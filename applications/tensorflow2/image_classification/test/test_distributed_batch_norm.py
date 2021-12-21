# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))


class TestDistributedBatchNorm(unittest.TestCase):

    def run_model_on_ipu(self, seed: int, replicas: int, distributed_batch_norm_replica_group_size: int):

        input_shape = (3, 3, 1)
        micro_batch_size = 5
        steps_per_execution = replicas * 2
        ds_size = micro_batch_size * steps_per_execution

        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = replicas
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
        cfg.norms.experimental.distributed_batch_norm_replica_group_size = (distributed_batch_norm_replica_group_size)
        cfg.configure_ipu_system()

        np.random.seed(seed)
        ds = tf.data.Dataset.from_tensor_slices(np.random.rand(ds_size, 3, 3, 1))
        ds = ds.map(lambda x: (x, (1,)))
        ds = ds.batch(micro_batch_size, drop_remainder=True)
        iterator = iter(ds)

        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        optimizer = tf.keras.optimizers.SGD()
        strategy = ipu.ipu_strategy.IPUStrategy()

        def validate(model):
            @tf.function
            def ipu_fn():
                image, _ = next(iterator)
                prediction = model(image, training=True)
                return prediction

            return strategy.run(ipu_fn)

        def train():
            with strategy.scope():

                input_layer = tf.keras.Input(shape=input_shape)
                x = input_layer
                x = tf.keras.layers.BatchNormalization(name='bn0')(x)
                x = tf.keras.layers.Flatten()(x)
                x = tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False,
                                          trainable=False, name='dense')(x)
                model = tf.keras.Model(input_layer, x)
                model.build(input_shape=(micro_batch_size, *input_shape))
                model.compile(optimizer=optimizer,
                              loss=loss_fn,
                              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                              steps_per_execution=steps_per_execution)
                model.fit(ds, steps_per_epoch=steps_per_execution, epochs=1)

            return model

        model = train()
        return validate(model)

    def test_distributed_batch_norm(self):
        pred_without_dist_batch_norm = self.run_model_on_ipu(
            seed=1, replicas=2, distributed_batch_norm_replica_group_size=1)
        pred_with_dist_batch_norm = self.run_model_on_ipu(
            seed=1, replicas=2, distributed_batch_norm_replica_group_size=2)
        assert pred_with_dist_batch_norm.numpy().sum() != pred_without_dist_batch_norm.numpy().sum()
