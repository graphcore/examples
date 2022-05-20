# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import time

import tensorflow as tf


class BatchStatisticsCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_nodes_processed_per_execution):
        self.num_nodes_processed_per_execution = num_nodes_processed_per_execution
        self.total_num_nodes_processed = 0

    def on_train_begin(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            batch_duration = time.time() - self.batch_start_time
            self.total_num_nodes_processed += self.num_nodes_processed_per_execution
            logs["throughput"] = self.num_nodes_processed_per_execution / batch_duration
            logs["total_num_nodes_processed"] = self.total_num_nodes_processed
