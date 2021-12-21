# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python import ipu
import unittest

from losses.loss_scaling import wrap_in_loss_scaling
from optimizers.l2_regularizer import add_l2_regularization


class TestOptimizerOptionsOnIPU(unittest.TestCase):

    @staticmethod
    def get_model():
        input_layer = tf.keras.Input(shape=1)
        x = tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False,
                                  trainable=True, name='dense')(input_layer)
        return tf.keras.Model(input_layer, x)

    def test_weight_decay(self):
        weight_decay = .1

        model = self.get_model()
        optimizer = tfa.optimizers.SGDW(weight_decay=weight_decay)

        grads = [tf.constant([0.], shape=(1, 1), dtype=tf.float32)]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        assert model.trainable_variables[0].numpy() == 1 - weight_decay

    def test_l2_regularization(self):
        l2_regularization = 0.1

        model = self.get_model()

        optimizer_class = tfa.optimizers.SGDW
        optimizer_class = add_l2_regularization(optimizer_class, l2_regularization)
        optimizer = optimizer_class(weight_decay=0, learning_rate=1.)

        grads = [tf.constant([0.], shape=(1, 1), dtype=tf.float32)]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        assert model.trainable_variables[0].numpy() == 1 - l2_regularization

    def test_loss_scaling(self):
        input_shape = (1, 1)
        micro_batch_size = 1

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        cfg.configure_ipu_system()

        ds = tf.data.Dataset.from_tensors(np.ones(input_shape))
        ds = ds.repeat()
        ds = ds.map(lambda x: (tf.cast(x, tf.float32), (0.001,)))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        iterator = iter(ds)
        loss_fn = tf.keras.losses.MSE
        optimizer_unscaled = tf.optimizers.SGD()
        optimizer_scaled = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer_unscaled,
            dynamic=False,
            initial_scale=9000
        )

        @tf.function
        def ipu_fn():

            with tf.GradientTape() as tape:
                image, label = next(iterator)
                prediction = model(image, training=True)
                loss = loss_fn(label, prediction)

            grads_unscaled = optimizer_unscaled.get_gradients(loss, model.trainable_weights)
            grads_scaled = optimizer_scaled.get_gradients(loss, model.trainable_weights)
            return grads_unscaled, grads_scaled

        strategy = ipu.ipu_strategy.IPUStrategy()

        with strategy.scope():

            input_layer = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Dense(1, kernel_initializer='zeros', use_bias=False,
                                      trainable=True, name='dense')(input_layer)
            model = tf.keras.Model(input_layer, x)

            model.build(input_shape=(micro_batch_size, input_shape))

            grads_unscaled, grads_scaled = strategy.run(ipu_fn)
            assert grads_unscaled[0].numpy()[0][0] != grads_scaled[0].numpy()[0][0]


class TestStaticLossScaling(unittest.TestCase):

    def ipu_prog(self, loss_scaling_factor: float, scale_gradient: bool, scale_loss: bool):

        micro_batch_size = 1

        ds = tf.data.Dataset.from_tensor_slices(([1.], [2.]))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        num_micro_batches = len(ds)

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        cfg.configure_ipu_system()

        strategy = ipu.ipu_strategy.IPUStrategy()
        with strategy.scope():
            input_layer = tf.keras.Input(shape=1)
            kernel_initializer = tf.keras.initializers.Constant(1)
            x = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(input_layer)
            model = tf.keras.Model(input_layer, x)

            model.build(input_shape=(micro_batch_size, 1))

            if scale_gradient:
                def gradient_normalizer(grads_and_vars): return [(
                    grad / loss_scaling_factor, var) for grad, var in grads_and_vars]
                optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, gradient_transformers=[gradient_normalizer])
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

            if scale_loss:
                loss = wrap_in_loss_scaling(loss_class=tf.keras.losses.MeanSquaredError,
                                            scaling_factor=loss_scaling_factor)()
            else:
                loss = tf.keras.losses.MeanSquaredError()

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=[tf.keras.losses.MeanSquaredError()],
                          steps_per_execution=num_micro_batches)

            model.fit(ds, steps_per_epoch=num_micro_batches)

            return model.get_weights()[0][0][0]

    def test_scale_gradient_only(self):
        weight_updated_with_scaled_gradient = self.ipu_prog(
            loss_scaling_factor=10, scale_gradient=True, scale_loss=False)
        weight_updated_without_scaled_gradient = self.ipu_prog(
            loss_scaling_factor=10, scale_gradient=False, scale_loss=False)
        assert weight_updated_without_scaled_gradient > weight_updated_with_scaled_gradient

    def test_loss_scale_only(self):
        weight_updated_with_scaled_loss = self.ipu_prog(loss_scaling_factor=10, scale_gradient=False, scale_loss=True)
        weight_updated_without_scaled_loss = self.ipu_prog(
            loss_scaling_factor=10, scale_gradient=False, scale_loss=False)
        assert weight_updated_with_scaled_loss > weight_updated_without_scaled_loss

    def test_loss_and_gradient_scaling(self):
        weight_with_loss_scaling = self.ipu_prog(loss_scaling_factor=10, scale_gradient=True, scale_loss=True)
        weight_without_loss_scaling = self.ipu_prog(loss_scaling_factor=10, scale_gradient=False, scale_loss=False)
        assert weight_with_loss_scaling == weight_without_loss_scaling
