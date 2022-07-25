# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from normalization import batch_norm
from optimizers.loss_scale_optimizer import add_loss_scaling_to_optimizer


class TestBatchNorm(unittest.TestCase):

    def run_model_on_ipu(self, ipu_batch_norm_layer=True, loss_scaling=None):

        input_shape = (3, 3, 1)
        micro_batch_size = 2

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        cfg.configure_ipu_system()

        ds = tf.data.Dataset.from_tensors(np.ones(input_shape))
        ds = ds.repeat()
        ds = ds.map(lambda x: (tf.cast(x, tf.float32), (1,)))
        ds = ds.batch(micro_batch_size, drop_remainder=True)

        iterator = iter(ds)
        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        optimizer_class = tf.keras.optimizers.SGD

        if loss_scaling:
            optimizer_class = add_loss_scaling_to_optimizer(optimizer_class, loss_scaling)

        if ipu_batch_norm_layer:
            optimizer_class = batch_norm.add_bn_moving_vars_updates_to_optimizer(
                optimizer_class, bn_momentum=0.99)

        optimizer = optimizer_class()

        @tf.function
        def ipu_fn():

            with tf.GradientTape() as tape:
                image, label = next(iterator)
                prediction = model(image, training=True)
                loss = loss_fn(label, prediction)

            optimizer.minimize(loss=loss, var_list=model.trainable_variables, tape=tape)
            return prediction

        strategy = ipu.ipu_strategy.IPUStrategy()

        with strategy.scope():

            input_layer = tf.keras.Input(shape=input_shape)
            x = input_layer
            if ipu_batch_norm_layer:
                x = batch_norm.BatchNormIPU(name='bn0')(x)
            else:
                x = tf.keras.layers.BatchNormalization(name='bn0')(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False, trainable=False, name='dense')(x)
            model = tf.keras.Model(input_layer, x)

            model.build(input_shape=(micro_batch_size, *input_shape))

        prediction = strategy.run(ipu_fn)

        return prediction, model

    def test_output_and_weights(self):
        prediction_vanilla_layer, vanilla_layer_model = self.run_model_on_ipu(ipu_batch_norm_layer=False)
        prediction_custom_layer, custom_layer_model = self.run_model_on_ipu(ipu_batch_norm_layer=True)

        # check if predictions are equal
        self.assertListEqual(prediction_custom_layer.numpy().tolist(), prediction_vanilla_layer.numpy().tolist())
        for vanilla_model_weight, custom_model_weight in zip(vanilla_layer_model.weights, custom_layer_model.weights):
            self.assertEqual(vanilla_model_weight.name, custom_model_weight.name)
            if 'bn0' in vanilla_model_weight.name:
                self.assertAlmostEqual(vanilla_model_weight.numpy()[0], custom_model_weight.numpy()[0])

    def test_combined_with_loss_scaling(self):
        _, vanilla_model = self.run_model_on_ipu(ipu_batch_norm_layer=False, loss_scaling=900)
        _, loss_scaled_model = self.run_model_on_ipu(ipu_batch_norm_layer=True, loss_scaling=900)

        for vanilla_model_weight, custom_model_weight in zip(vanilla_model.weights, loss_scaled_model.weights):
            self.assertEqual(vanilla_model_weight.name, custom_model_weight.name)
            if 'bn0' in vanilla_model_weight.name:
                self.assertAlmostEqual(vanilla_model_weight.numpy()[0], custom_model_weight.numpy()[0])
