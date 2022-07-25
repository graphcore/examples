# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf


class CorrectionMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric_name: str, correction_factor: int):
        self.metric_name = metric_name
        self.correction_factor = correction_factor

    def on_train_batch_end(self, batch, logs=None):
        self.correct_metric(logs)

    def on_test_batch_end(self, batch, logs=None):
        self.correct_metric(logs)

    def correct_metric(self, logs=None):
        if logs is not None:
            if self.metric_name in logs:
                logs[self.metric_name] *= self.correction_factor
