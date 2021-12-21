# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import wandb
import datetime
import tensorflow as tf
from custom_exceptions import MissingArgumentException
from .periodic_metrics import PeriodicMetrics

ONE_OFF_METRICS = ['Compilation Time']


class CustomWandbCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_period: int, args: dict, model: tf.keras.Model):
        self.log_period = log_period
        self.model = model
        self.__current_batch_operations = self.__first_batch_operations
        self.initialise_wandb(args)

    def initialise_wandb(self, args: dict):
        project = 'TF2-classification'
        if 'model_name' not in args.keys():
            raise MissingArgumentException('Argument \'model_name\' is missing for W&B.')
        if 'dataset' not in args.keys():
            raise MissingArgumentException('Argument \'dataset\' is missing for W&B.')
        name = f"{args['model_name']}-{args['dataset']}-{str(datetime.datetime.now())}"
        wandb.init(entity='sw-apps', project=project, name=name, config=args)

    def on_train_begin(self, logs=None):
        wandb.run.summary['graph'] = wandb.Graph.from_keras(self.model)

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

        one_off_metrics_logs = {metric: logs[metric] for metric in logs if metric in ONE_OFF_METRICS}
        if one_off_metrics_logs:
            wandb.log(one_off_metrics_logs)

        # filter one off metrics
        logs = {metric: logs[metric] for metric in logs if metric not in ONE_OFF_METRICS}

        # which metrics are accumulated are only known at runtime
        # but stay the same for the duration of training
        self.metrics = PeriodicMetrics(list(logs.keys()))

        # this needs to be called so we don't discard the metrics of the first batch
        self.__next_batches_operations(logs)
        self.__current_batch_operations = self.__next_batches_operations
