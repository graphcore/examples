# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf


class EpochCalculationCallback(tf.keras.callbacks.Callback):
    def __init__(self, micro_batches_per_epoch: int):
        self.epoch = 0
        self.micro_batches_per_epoch = micro_batches_per_epoch
        self.prev_batch = 0

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            logs["epoch"] = self.epoch + (batch + 1) / self.micro_batches_per_epoch

    def on_epoch_begin(self, epoch, _):
        self.epoch = epoch

    def on_test_batch_end(self, batch, logs=None):
        raise ValueError("use an EpochFromCkptNameCallback callback instead")
