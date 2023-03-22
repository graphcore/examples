# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf


class LossScalingFactorLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, auto_loss_scaling, name="loss_scaling_factor"):
        super().__init__()
        self.name = name
        self.auto_loss_scaling = auto_loss_scaling

    def on_train_batch_end(self, _, logs=None):
        if logs is not None:
            logs[self.name] = self.get_loss_scaling_factor()

    def on_epoch_end(self, _, logs=None):
        if logs is not None:
            logs[self.name] = self.get_loss_scaling_factor()

    def get_loss_scaling_factor(self) -> float:
        if self.auto_loss_scaling:
            loss_scaling_factor = tf.keras.backend.get_value(self.model.optimizer.loss_scaling_factor)
            if isinstance(loss_scaling_factor, tf.Tensor):
                return loss_scaling_factor.numpy()
            else:
                return loss_scaling_factor
        else:
            pass
