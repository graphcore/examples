# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import logging
from .periodic_metrics import PeriodicMetrics

ONE_OFF_METRICS = ["Compilation Time"]
units = {
    "throughput": "samples/sec",
    "%CPU memory": "%",
    "Compilation Time": "s",
    "validation_accuracy": "%",
    "training_accuracy": "%",
}


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_period: int):
        self.log_period = log_period
        self.__current_batch_operations = self.__first_batch_operations
        self.logger = logging.getLogger("logging_callback")
        self.logger.info(f"logging every {log_period} micro batches")

    def on_train_batch_end(self, batch, logs=None):
        self.log_data(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.log_data(batch, logs)

    def log_data(self, batch, logs=None):
        self.__current_batch_operations(logs)

        if (batch + 1) % self.log_period == 0:
            metrics = self.metrics.get_normalized()

            metrics_print = []
            for metric in metrics:
                if metric in units:
                    metrics_print.append(f"{metric}: {metrics[metric]} {units[metric]}")
                else:
                    metrics_print.append(f"{metric}: {metrics[metric]} ")

            self.logger.info(f"batch {batch+1}: {metrics_print}")
            self.metrics.reset()

    def __next_batches_operations(self, logs):
        self.metrics.update(logs)

    def __first_batch_operations(self, logs):

        for key in logs:
            if key in ONE_OFF_METRICS:
                if key in units:
                    self.logger.info(f"{key} {logs[key]} {units[key]}")
                else:
                    self.logger.info(f"{key} {logs[key]} ")

        # filter one off metrics
        logs = {metric: logs[metric] for metric in logs if metric not in ONE_OFF_METRICS}

        # which metrics are accumulated are only known at runtime
        # but stay the same for the duration of training
        self.metrics = PeriodicMetrics(list(logs.keys()))

        # this needs to be called so we don't discard the metrics of the first batch
        self.__next_batches_operations(logs)
        self.__current_batch_operations = self.__next_batches_operations
