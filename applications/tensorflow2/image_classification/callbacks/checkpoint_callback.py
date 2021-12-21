# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import os


class CheckpointCallback(tf.keras.callbacks.Callback):

    def __init__(self, ckpt_period: int, checkpoint_dir: str):
        self.ckpt_period = ckpt_period
        self.global_batch_counter = 0
        self.checkpoint_dir = checkpoint_dir
        self.epochs_since_last_save = 0
        if not (os.path.exists(self.checkpoint_dir)):
            os.makedirs(self.checkpoint_dir)

    def on_train_begin(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        self.global_batch_counter = 0

    def on_train_batch_end(self, batch, logs=None):

        if (batch + 1) % self.ckpt_period == 0:
            filepath = os.path.join(self.checkpoint_dir,
                                    f'_epoch_{self.epochs_since_last_save}_global_batch_{self.global_batch_counter}.h5')
            self.model.save_weights(filepath)

        self.global_batch_counter += 1
