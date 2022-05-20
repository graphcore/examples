# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging

import tensorflow as tf

from keras_extensions.callbacks.periodic_metrics import PeriodicMetrics

LOGGING_ONE_OFF_METRICS = [""]
LOGGING_EXCLUDE_METRICS = ["total_num_nodes_processed"]


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_period: int):
        self.log_period = log_period
        self.__current_batch_operations = self.__first_batch_operations
        self.logger = logging.getLogger("logging_callback")
        self.logger.info("Logging with a period of 1 log every"
                         f" {log_period} executions.")
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1

    def on_train_batch_end(self, batch, logs=None):
        self.log_data(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.log_data(batch, logs)

    def log_data(self, batch, logs=None):
        self.__current_batch_operations(logs)

        if (batch + 1) % self.log_period == 0:
            metrics = self.metrics.get_normalized()
            logger_metrics_str = ", ".join(
                [f"{k}: {v:.3f}" for k, v in metrics.items() if k not in LOGGING_EXCLUDE_METRICS])
            self.logger.info(f"Epoch {self.epoch} - Batch {batch+1}: {logger_metrics_str}")
            self.metrics.reset()

    def __next_batches_operations(self, logs):
        self.metrics.update(logs)

    def __first_batch_operations(self, logs):

        for key in logs:
            if key in LOGGING_ONE_OFF_METRICS:
                self.logger.info(f"{key} {logs[key]}")

        # filter one off metrics
        logs = {metric: logs[metric]
                for metric in logs if metric not in LOGGING_ONE_OFF_METRICS}

        # which metrics are accumulated are only known at runtime
        # but stay the same for the duration of training
        self.metrics = PeriodicMetrics(list(logs.keys()))

        # this needs to be called so we don't discard the metrics of the first batch
        self.__next_batches_operations(logs)
        self.__current_batch_operations = self.__next_batches_operations
