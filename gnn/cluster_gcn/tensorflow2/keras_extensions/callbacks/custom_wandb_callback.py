# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import wandb

from keras_extensions.callbacks.periodic_metrics import PeriodicMetrics

VALIDATION_EXCLUDE_METRICS = ["throughput", "total_num_nodes_processed"]


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 name: str,
                 config: dict):
        self.__current_batch_operations = self.__first_batch_operations


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


class TrainingCustomWandbCallback(CustomWandbCallback):

    def on_train_batch_end(self, batch, logs=None):
        self.upload_to_wandb(batch, logs)

    def on_train_end(self, logs=None):
        if logs is not None:
            wandb.log({"mean_tput": logs.get("mean_throughput", "nan"),
                       "STD_tput": logs.get("std_throughput", "nan"),
                       "mean_real_tput": logs.get("mean_real_throughput", "nan")})


class ValidationCustomWandbCallback(CustomWandbCallback):

    def on_test_batch_end(self, batch, logs=None):
        val_logs = {k+"_validation": logs[k] for k in logs.keys()}
        self.upload_to_wandb(batch, val_logs)

    def __first_batch_operations(self, logs):
        logs = {k+"_validation": logs[k] for k in logs.keys() if k not in VALIDATION_EXCLUDE_METRICS}
        return super().__first_batch_operations(logs)
