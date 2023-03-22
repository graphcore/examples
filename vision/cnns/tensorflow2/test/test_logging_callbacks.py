# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).absolute().parent.parent))
from callbacks.logging_callback import LoggingCallback
import logging
from tensorflow.python import ipu
import re


class TestLoggingCallbacks(unittest.TestCase):
    def test_compilation_time_logging(self):
        with self.assertLogs("logging_callback", level=logging.INFO) as test_log:
            callback = LoggingCallback(log_period=10)

            callback.on_train_begin()
            callback.on_train_batch_begin(batch=0)
            callback.on_train_batch_end(batch=0, logs={"Compilation Time": 30})

            self.assertEqual(len(test_log.output), 2)
            self.assertIn("logging every 10 micro batches", test_log.output[0])
            self.assertIn("Compilation Time 30", test_log.output[1])

    def test_log_period(self):
        with self.assertLogs("logging_callback", level=logging.INFO) as test_log:
            log_period = 7
            callback = LoggingCallback(log_period)

            callback.on_train_begin()
            self.assertIn(f"logging every {log_period} micro batches", " ".join(test_log.output))

            for batch in range(30):

                callback.on_train_batch_begin(batch)
                callback.on_train_batch_end(batch, logs={"throughput": 0})

                test_log_output_str = " ".join(test_log.output)
                search_str = f"batch {batch+1}: ['throughput:"
                if (batch + 1) % log_period == 0:
                    self.assertIn(search_str, test_log_output_str)
                else:
                    self.assertNotIn(search_str, test_log_output_str)

    def run_ipu_prog(self, callbacks, num_steps):

        weight_value = 2.0
        strategy = ipu.ipu_strategy.IPUStrategy()

        ds = tf.data.Dataset.from_tensor_slices(([1.0, 1.0], [1.0, 2.0]))
        ds = ds.batch(1, drop_remainder=True)
        ds = ds.repeat()

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.device_connection.enable_remote_buffers = True
        cfg.configure_ipu_system()

        with strategy.scope():
            optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
            input_layer = tf.keras.Input(shape=1)
            kernel_initializer = tf.keras.initializers.Constant(weight_value)
            x = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(input_layer)
            model = tf.keras.Model(input_layer, x)
            model.build(input_shape=(1, 1))
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE, steps_per_execution=num_steps)
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=1)
            model.fit(ds, steps_per_epoch=num_steps, callbacks=callbacks)
            return model.get_weights()[0][0][0]

    def test_average_log(self):
        with self.assertLogs("logging_callback", level=logging.INFO) as test_log:
            num_steps = 2
            callback = LoggingCallback(log_period=num_steps)
            last_weigth = self.run_ipu_prog(callback, num_steps)
            print(test_log)
            self.assertEqual(last_weigth, 4.0)
            self.assertIn("loss: 2.5", test_log.output[1])
