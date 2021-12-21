# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import unittest
import tensorflow as tf
from tensorflow.python import ipu


class TestGradientNormalization(unittest.TestCase):

    def ipu_prog(self, num_elements, num_replicas, gradient_accumulation):

        micro_batch_size = int(num_elements / num_replicas / gradient_accumulation)

        ds = tf.data.Dataset.from_tensor_slices(([1.] * num_elements, [2.] * num_elements))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        num_micro_batches = len(ds)

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = num_replicas
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        cfg.configure_ipu_system()

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            input_layer = tf.keras.Input(shape=1)
            kernel_initializer = tf.keras.initializers.Constant(1)
            x = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(input_layer)
            model = tf.keras.Model(input_layer, x)

            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation)
            model.build(input_shape=(micro_batch_size, 1))

            def gradient_normalizer(grads_and_vars): return [(
                grad / num_replicas / gradient_accumulation, var) for grad, var in grads_and_vars]

            optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, gradient_transformers=[gradient_normalizer])
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE, metrics=[
                          tf.keras.losses.MSE], steps_per_execution=num_micro_batches)
            model.fit(ds, steps_per_epoch=num_micro_batches)

            return model.get_weights()[0][0][0]

    def test_1replica_1grad_acc(self):
        weight = (self.ipu_prog(4, 1, 1))
        self.assertEqual(weight, 3.)

    def test_2replica_1grad_acc(self):
        weight = self.ipu_prog(4, 2, 1)
        self.assertEqual(weight, 3.)

    def test_1replica_2grad_acc(self):
        weight = self.ipu_prog(4, 1, 2)
        self.assertEqual(weight, 3.)

    def test_2replica_2grad_acc(self):
        weight = self.ipu_prog(4, 2, 2)
        self.assertEqual(weight, 3.)

    def test_microbatch_replicas_grad_acc(self):
        weight = self.ipu_prog(8, 2, 2)
        self.assertEqual(weight, 3.)
