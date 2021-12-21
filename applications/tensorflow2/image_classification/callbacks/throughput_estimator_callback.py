# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import time


class ThroughputEstimatorCallback(tf.keras.callbacks.Callback):
    def __init__(self, steps_per_execution: int, micro_batch_size: int):
        self.steps_per_execution = steps_per_execution
        self.micro_batch_size = micro_batch_size

    def on_train_begin(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            batch_duration = time.time() - self.batch_start_time
            logs['Average images/s'] = self.steps_per_execution * self.micro_batch_size / batch_duration

    def on_test_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_test_batch_end(self, batch, logs=None):
        if logs is not None:
            batch_duration = time.time() - self.batch_start_time
            logs['Average images/s'] = self.steps_per_execution * self.micro_batch_size / batch_duration
