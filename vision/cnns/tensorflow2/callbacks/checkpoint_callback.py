# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import os


class CheckpointCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 ckpt_period: int,
                 ckpt_phase: int,
                 checkpoint_dir: str):

        self.ckpt_period = ckpt_period
        self.ckpt_phase = ckpt_phase
        self.batch_counter = 0
        self.prev_batch = 0
        self.epoch_counter = 0
        self.checkpoint_dir = checkpoint_dir
        if not (os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_counter = epoch
        self.prev_batch = 0

    def on_train_batch_end(self, batch, logs=None):

        batches_between_calls = (batch+1) - self.prev_batch
        self.prev_batch = (batch+1)
        self.batch_counter += batches_between_calls
        if (self.batch_counter) % self.ckpt_period == self.ckpt_phase:
            epoch = logs['epoch'] if 'epoch' in logs else self.epoch_counter
            filepath = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.h5')
            self.model.save_weights(filepath)

