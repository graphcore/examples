# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import time


class ThroughputEstimatorCallback(tf.keras.callbacks.Callback):
    def __init__(self, images_per_execution: int):
        self.images_per_execution = images_per_execution

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.calc_tput(logs)

    def on_test_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_test_batch_end(self, batch, logs=None):
        self.calc_tput(logs)

    def calc_tput(self, logs=None):
        if logs is not None:
            batch_duration = time.time() - self.batch_start_time
            logs['Average images/s'] = self.images_per_execution / batch_duration
