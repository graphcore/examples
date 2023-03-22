# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf


class LossRescalerCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_scaling: float):
        """Rescaling the loss to correct value."""
        self.loss_scaling = loss_scaling

    def on_train_batch_end(self, _, logs=None):
        logs["loss"] /= self.loss_scaling
