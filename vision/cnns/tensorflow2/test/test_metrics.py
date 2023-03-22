# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
import unittest
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from losses import loss_enqueuer
from metrics import metric_enqueuer
from callbacks import outfeed_queue_callback, logging_callback


class ValidationMetricsTest(unittest.TestCase):
    def get_model(self):
        input_layer = tf.keras.Input(shape=1)
        x = tf.keras.layers.Dense(1, kernel_initializer="zeros", use_bias=False, trainable=False, name="dense")(
            input_layer
        )
        return tf.keras.Model(input_layer, x)

    def metric_prog(self, steps_per_execution, num_replicas):
        input_shape = (1, 1)
        micro_batch_size = 1

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = num_replicas
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()

        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 2.0, 3.0, 4.0]
        ys_pred = [0.0, 0.0, 0.0, 0.0]
        expected_error = tf.keras.metrics.MSE(np.array(ys_pred), np.array(ys)).numpy()

        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            model = self.get_model()
            model.build(input_shape=(micro_batch_size, *input_shape))
            model.compile(metrics=tf.keras.metrics.MSE, steps_per_execution=steps_per_execution)
            score = model.evaluate(ds, steps=4)

        return score[-1], expected_error

    def test_metric_1_steps(self):
        value, expected_value = self.metric_prog(steps_per_execution=1, num_replicas=1)
        self.assertEqual(value, expected_value)

    def test_metric_2_steps(self):
        value, expected_value = self.metric_prog(steps_per_execution=2, num_replicas=1)
        self.assertEqual(value, expected_value)

    def test_metric_4_steps(self):
        value, expected_value = self.metric_prog(steps_per_execution=4, num_replicas=1)
        self.assertEqual(value, expected_value)

    def test_metric_2_steps_2_replicas(self):
        value, expected_value = self.metric_prog(steps_per_execution=1, num_replicas=2)
        # Note that it should be equal, but it is currently bugged.
        # When this test fails then the bug was fixed
        self.assertNotEqual(value, expected_value)

    def test_metric_4_steps_2_replicas(self):
        value, expected_value = self.metric_prog(steps_per_execution=2, num_replicas=2)
        # Note that it should be equal, but it is currently bugged.
        # When this test fails then the bug was fixed
        self.assertNotEqual(value, expected_value)


class TrainMetricsTest(unittest.TestCase):
    def metric_prog(self, num_elements, micro_batch_size, num_replicas, gradient_accumulation, steps_per_execution):

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = num_replicas
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()

        xs = list(range(num_elements))
        ys = list(range(num_elements))
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            input_layer = tf.keras.Input(shape=1)
            x = tf.keras.layers.Dense(1, kernel_initializer="zeros", use_bias=False, name="dense")(input_layer)
            model = tf.keras.Model(input_layer, x)
            model.build(input_shape=(micro_batch_size, 1))
            optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
            model.compile(optimizer=optimizer, loss=tf.keras.metrics.MSE, steps_per_execution=steps_per_execution)
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation)
            history = model.fit(ds, steps_per_epoch=num_elements // micro_batch_size)

        return history

    def test_4micro_batch_size_1_step_per_execution(self):

        history = self.metric_prog(
            num_elements=4, micro_batch_size=4, num_replicas=1, gradient_accumulation=1, steps_per_execution=1
        )

        self.assertEqual(history.history["loss"][0], 3.5)

    def test_1micro_batch_size_4_replicas(self):

        history = self.metric_prog(
            num_elements=4, micro_batch_size=1, num_replicas=4, gradient_accumulation=1, steps_per_execution=1
        )

        # Note that it should be equal, but it is currently bugged.
        # When this test fails then the bug was fixed
        self.assertNotEqual(history.history["loss"][0], 3.5)

    def test_1micro_batch_size_4_gradient_accumulation(self):

        history = self.metric_prog(
            num_elements=4, micro_batch_size=1, num_replicas=1, gradient_accumulation=4, steps_per_execution=4
        )

        # Note that it should be equal, but it is currently bugged.
        # When this test fails then the bug was fixed
        self.assertNotEqual(history.history["loss"][0], 3.5)

    def test_1micro_batch_size_2_replicas_2_gradient_accumulation(self):

        history = self.metric_prog(
            num_elements=4, micro_batch_size=1, num_replicas=2, gradient_accumulation=2, steps_per_execution=2
        )

        # Note that it should be equal, but it is currently bugged.
        # When this test fails then the bug was fixed
        self.assertNotEqual(history.history["loss"][0], 3.5)


class EnqueuedLossTest(unittest.TestCase):
    def ipu_prog(self, num_elements, micro_batch_size, num_replicas, gradient_accumulation, steps_per_execution):

        # configure logger
        logging.basicConfig(level=logging.INFO)

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = num_replicas
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()

        xs = list(range(num_elements))
        ys = list(range(num_elements))
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            input_layer = tf.keras.Input(shape=1)
            x = tf.keras.layers.Dense(1, kernel_initializer="zeros", use_bias=False, name="dense")(input_layer)
            model = tf.keras.Model(input_layer, x)
            model.build(input_shape=(micro_batch_size, 1))
            optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
            loss_class = tf.keras.losses.MeanSquaredError
            loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            loss_class = loss_enqueuer.wrap_loss_in_enqueuer(loss_class, loss_outfeed_queue)
            loss = loss_class()
            model.compile(optimizer=optimizer, loss=loss, steps_per_execution=steps_per_execution)
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation)
            callbacks = [
                outfeed_queue_callback.OutFeedQueueCallback(loss_outfeed_queue, "average loss"),
                logging_callback.LoggingCallback(gradient_accumulation),
            ]
            model.fit(ds, steps_per_epoch=num_elements // micro_batch_size, callbacks=callbacks)

    def test_4micro_batch_size_1step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=4, num_replicas=1, gradient_accumulation=1, steps_per_execution=1
            )
            print(test_log.output)
            self.assertIn("average loss: 3.5", " ".join(test_log.output))

    def test_2micro_batch_size_2replicas_1step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=2, num_replicas=2, gradient_accumulation=1, steps_per_execution=1
            )
            self.assertIn("average loss: 3.5", " ".join(test_log.output))

    def test_1micro_batch_size_4replicas_1step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=1, num_replicas=4, gradient_accumulation=1, steps_per_execution=1
            )
            self.assertIn("average loss: 3.5", " ".join(test_log.output))

    def test_1micro_batch_size_4gradientacc_4step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=1, num_replicas=1, gradient_accumulation=4, steps_per_execution=4
            )
            self.assertIn("average loss: 3.5", " ".join(test_log.output))

    def test_1micro_batch_size_2replicas_2gradientacc_2step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=1, num_replicas=2, gradient_accumulation=2, steps_per_execution=2
            )
            self.assertIn("average loss: 3.5", " ".join(test_log.output))


class EnqueuedMetricTest(unittest.TestCase):
    def ipu_prog(self, num_elements, micro_batch_size, num_replicas, gradient_accumulation, steps_per_execution):

        # configure logger
        logging.basicConfig(level=logging.INFO)

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = num_replicas
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()

        xs = list(range(num_elements))
        ys = list(range(num_elements))
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            input_layer = tf.keras.Input(shape=1)
            x = tf.keras.layers.Dense(1, kernel_initializer="zeros", use_bias=False, name="dense")(input_layer)
            model = tf.keras.Model(input_layer, x)
            model.build(input_shape=(micro_batch_size, 1))
            optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
            loss = tf.keras.losses.MeanSquaredError()
            metric_class = tf.keras.metrics.MeanSquaredError
            metric_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            metric_class = metric_enqueuer.wrap_metric_in_enqueuer(metric_class, metric_outfeed_queue)
            metric = metric_class()
            model.compile(optimizer=optimizer, loss=loss, metrics=[metric], steps_per_execution=steps_per_execution)
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation)
            callbacks = [
                outfeed_queue_callback.OutFeedQueueCallback(metric_outfeed_queue, "average metric"),
                logging_callback.LoggingCallback(steps_per_execution),
            ]
            model.fit(ds, steps_per_epoch=num_elements // micro_batch_size, callbacks=callbacks)

    def test_4micro_batch_size_1step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=4, num_replicas=1, gradient_accumulation=1, steps_per_execution=1
            )
            print(test_log.output)
            self.assertIn("average metric: 3.5", " ".join(test_log.output))

    def test_2micro_batch_size_2replicas_1step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=2, num_replicas=2, gradient_accumulation=1, steps_per_execution=1
            )
            self.assertIn("average metric: 3.5", " ".join(test_log.output))

    def test_1micro_batch_size_4replicas_1step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=1, num_replicas=4, gradient_accumulation=1, steps_per_execution=1
            )
            self.assertIn("average metric: 3.5", " ".join(test_log.output))

    def test_1micro_batch_size_4gradientacc_4step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=1, num_replicas=1, gradient_accumulation=4, steps_per_execution=4
            )
            self.assertIn("average metric: 3.5", " ".join(test_log.output))

    def test_1micro_batch_size_2replicas_2gradientacc_2step_per_execution(self):

        with self.assertLogs() as test_log:
            self.ipu_prog(
                num_elements=4, micro_batch_size=1, num_replicas=2, gradient_accumulation=2, steps_per_execution=2
            )
            self.assertIn("average metric: 3.5", " ".join(test_log.output))
