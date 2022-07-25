# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf


class LRLoggerCallback(tf.keras.callbacks.Callback):

    def __init__(self, name='lr'):
        super().__init__()
        self.name = name

    def on_train_batch_end(self, _, logs=None):
        if logs is not None:
            logs[self.name] = self.get_lr()

    def on_epoch_end(self, _, logs=None):
        if logs is not None:
            logs[self.name] = self.get_lr()

    def get_lr(self) -> float:
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return lr(self.model.optimizer.iterations).numpy()
        elif isinstance(lr, tf.Tensor):
            return lr.numpy()
        else:
            return lr
