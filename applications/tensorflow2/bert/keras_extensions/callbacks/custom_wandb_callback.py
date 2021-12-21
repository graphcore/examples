# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import datetime

import tensorflow as tf
import wandb

from keras_extensions.callbacks.periodic_metrics import PeriodicMetrics


WANDB_ONE_OFF_METRICS = ['Compilation Time']


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 name: str,
                 log_period: int,
                 config: dict,
                 model: tf.keras.Model):
        self.log_period = log_period
        self.model = model
        self.__current_batch_operations = self.__first_batch_operations
        self.initialise_wandb(name, config)

    def initialise_wandb(self, name: str, config: dict):
        wandb.init(entity="sw-apps",
                   project="TF2-BERT",
                   name=name,
                   config=config,
                   tags=config["wandb_opts"]["init"]["tags"])

    def on_train_batch_end(self, batch, logs=None):
        self.upload_to_wandb(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.upload_to_wandb(batch, logs)

    def upload_to_wandb(self, batch, logs=None):
        self.__current_batch_operations(logs)

        if (batch + 1) % self.log_period == 0:
            metrics = self.metrics.get_normalized()
            wandb.log(metrics)

            self.metrics.reset()

    def __next_batches_operations(self, logs):
        self.metrics.update(logs)

    def __first_batch_operations(self, logs):

        one_off_metrics_logs = {metric: logs[metric] for metric in logs if metric in WANDB_ONE_OFF_METRICS}
        if one_off_metrics_logs:
            wandb.log(one_off_metrics_logs)

        # filter one off metrics
        logs = {metric: logs[metric] for metric in logs if metric not in WANDB_ONE_OFF_METRICS}

        # which metrics are accumulated are only known at runtime
        # but stay the same for the duration of training
        self.metrics = PeriodicMetrics(list(logs.keys()))

        # this needs to be called so we don't discard the metrics of the first batch
        self.__next_batches_operations(logs)
        self.__current_batch_operations = self.__next_batches_operations
