# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow.python import ipu

sys.path.append(str(Path(__file__).absolute().parent.parent))
from losses.smoothed_categorical_crossentropy import SmoothedCategoricalCrossentropy


class TestLossesIdentical(unittest.TestCase):
    def ipu_prog(
        self, num_elements, loss, micro_batch_size=2, num_replicas=1, gradient_accumulation=2, steps_per_execution=4
    ):

        weight_value = 160.0

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
            kernel_initializer = tf.keras.initializers.Constant(weight_value)
            input_layer = tf.keras.Input(shape=1)
            x = tf.keras.layers.Dense(
                num_elements, kernel_initializer=kernel_initializer, use_bias=False, name="dense", activation="softmax"
            )(input_layer)
            model = tf.keras.Model(input_layer, x)
            model.build(input_shape=(micro_batch_size, 1))
            optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
            model.compile(optimizer=optimizer, loss=loss, steps_per_execution=steps_per_execution)
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation)
            history = model.fit(ds, steps_per_epoch=num_elements // micro_batch_size)
            return history.history["loss"][-1]

    def test_smoothing_zero(self):
        num_elements = 8
        loss1 = self.ipu_prog(num_elements=num_elements, loss=tf.keras.losses.SparseCategoricalCrossentropy())
        loss2 = self.ipu_prog(num_elements=num_elements, loss=SmoothedCategoricalCrossentropy(num_elements, 0.0))
        assert loss1 == loss2
