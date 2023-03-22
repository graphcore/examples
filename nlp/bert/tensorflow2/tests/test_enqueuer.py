# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import tensorflow as tf
from tensorflow.python import ipu

from utilities.metric_enqueuer import wrap_loss_in_enqueuer


def ipu_prog(wrapped_loss, heads=1):
    micro_batch_size = 2

    ds = tf.data.Dataset.from_tensor_slices(([1.0] * micro_batch_size, [2.0] * micro_batch_size))
    ds = ds.batch(micro_batch_size, drop_remainder=True)

    num_micro_batches = len(ds)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        input_layer = tf.keras.Input(shape=1)
        kernel_initializer = tf.keras.initializers.Constant(1)

        outputs = []
        for i in range(heads):
            outputs.append(
                tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer, name=f"dense_{i}")(
                    input_layer
                )
            )

        model = tf.keras.Model(input_layer, outputs)
        model.build(input_shape=(micro_batch_size, 1))
        model.compile(optimizer="SGD", loss=wrapped_loss, steps_per_execution=num_micro_batches)
        return model.fit(ds, steps_per_epoch=num_micro_batches)


class TestLossEnqueuer:
    def test_single_loss_enqueued(self):
        loss_key = "loss"
        loss = wrap_loss_in_enqueuer(tf.keras.losses.CategoricalCrossentropy, [loss_key])()
        history = ipu_prog(loss)

        assert loss.outfeed_queue.enqueued
        x = loss.outfeed_queue.dequeue()
        assert loss_key in x
        np.testing.assert_allclose(history.history[loss_key][0], x[loss_key][0][-1])

    def test_multiple_loss_enqueued_from_single_loss_function(self):
        loss_key_1 = "loss_1"
        loss_key_2 = "loss_2"
        loss = wrap_loss_in_enqueuer(tf.keras.losses.CategoricalCrossentropy, [loss_key_1, loss_key_2])()
        history = ipu_prog({"dense_0": loss, "dense_1": loss}, heads=2)

        assert loss.outfeed_queue.enqueued
        x = loss.outfeed_queue.dequeue()
        assert loss_key_1 in x
        assert loss_key_2 in x
        np.testing.assert_allclose(history.history["dense_0_loss"][0], x[loss_key_1][0][-1])
        np.testing.assert_allclose(history.history["dense_1_loss"][0], x[loss_key_2][0][-1])

    def test_multiple_loss_enqueued_from_multiple_loss_functions(self):
        loss_key_1 = "loss_1"
        loss_1 = wrap_loss_in_enqueuer(tf.keras.losses.CategoricalCrossentropy, [loss_key_1])()
        loss_key_2 = "loss_2"
        loss_2 = wrap_loss_in_enqueuer(tf.keras.losses.CategoricalCrossentropy, [loss_key_2])()
        history = ipu_prog({"dense_0": loss_1, "dense_1": loss_2}, heads=2)

        assert loss_1.outfeed_queue.enqueued
        x = loss_1.outfeed_queue.dequeue()
        assert loss_key_1 in x
        np.testing.assert_allclose(history.history["dense_0_loss"][0], x[loss_key_1][0][-1])

        assert loss_2.outfeed_queue.enqueued
        x = loss_2.outfeed_queue.dequeue()
        assert loss_key_2 in x
        np.testing.assert_allclose(history.history["dense_1_loss"][0], x[loss_key_2][0][-1])
