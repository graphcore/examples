# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import wandb

from keras_extensions.callbacks.periodic_metrics import PeriodicMetrics


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 name: str,
                 config: dict):
        self.__current_batch_operations = self.__first_batch_operations

    def on_train_batch_end(self, batch, logs=None):
        self.upload_to_wandb(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.upload_to_wandb(batch, logs)

    def upload_to_wandb(self, batch, logs=None):
        self.__current_batch_operations(logs)
        metrics = self.metrics.get_normalized()
        wandb.log({k: v for k, v in metrics.items()})
        self.metrics.reset()

    def __next_batches_operations(self, logs):
        self.metrics.update(logs)

    def __first_batch_operations(self, logs):

        logs = {metric: logs[metric] for metric in logs}
        self.metrics = PeriodicMetrics(list(logs.keys()))

        # this needs to be called so we don't discard the metrics of the first batch
        self.__next_batches_operations(logs)
        self.__current_batch_operations = self.__next_batches_operations
