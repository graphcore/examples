# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import time

import tensorflow as tf
import numpy as np


class BatchStatisticsCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_nodes_processed_per_execution, real_over_padded_ratio, total_num_epochs, loss):
        self.num_nodes_processed_per_execution = num_nodes_processed_per_execution
        self.real_over_padded_ratio = real_over_padded_ratio
        self.total_num_epochs = total_num_epochs
        self.total_num_nodes_processed = 0
        self.throughput_logs = []
        self.loss = loss

    def on_train_begin(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            batch_duration = time.time() - self.batch_start_time
            self.total_num_nodes_processed += self.num_nodes_processed_per_execution
            logs["throughput"] = self.num_nodes_processed_per_execution / batch_duration
            self.throughput_logs.append(logs["throughput"])
            logs["total_num_nodes_processed"] = self.total_num_nodes_processed

    def on_train_end(self, logs=None):
        if logs is not None:
            steps_to_skip = min(len(self.throughput_logs) - 1, 6)
            logs["mean_throughput"] = np.mean(self.throughput_logs[steps_to_skip:])
            logs["std_throughput"] = np.std(self.throughput_logs[steps_to_skip:])
            logs["mean_real_throughput"] = logs["mean_throughput"] * self.real_over_padded_ratio
            logs["loss"] = self.loss
