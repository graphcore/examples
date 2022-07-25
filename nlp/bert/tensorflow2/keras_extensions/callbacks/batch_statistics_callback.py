# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import time

import tensorflow as tf

from data_utils.batch_config import BatchConfig


class BatchStatisticsCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_config: BatchConfig):
        self.batch_config = batch_config
        self.total_num_samples_processed = 0

    def on_train_begin(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            batch_duration = time.time() - self.batch_start_time
            self.total_num_samples_processed += self.batch_config.num_samples_processed_per_execution
            logs["throughput"] = self.batch_config.num_samples_processed_per_execution / batch_duration
            logs["num_samples"] = self.total_num_samples_processed
