# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import numpy as np


class OptimizationMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_field, target_value) -> None:
        self.target_field = target_field
        self.target_value = target_value
        self.margin = 0.02
        self.optimization_metric = None

    def on_test_batch_end(self, batch, logs=None):
        assert self.target_field in logs
        assert "epoch" in logs

        err = np.minimum(logs[self.target_field] - self.target_value, 0)
        self.optimization_metric = np.exp(-0.5 * (err) ** 2 / (self.margin**2)) / logs["epoch"]
