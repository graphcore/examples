# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import datetime
from enum import Enum
import multiprocessing
import os

import numpy as np
import pandas as pd
import argparse
import time
import tensorflow as tf


def exp_rmspe(y_true, y_pred):
    """Competition evaluation metric, expects logarithic inputs."""
    pct = tf.square((tf.exp(y_true) - tf.exp(y_pred)) / tf.exp(y_true))
    # Compute mean excluding stores with zero denominator.
    x = tf.reduce_sum(tf.where(y_true > 0.001, pct, tf.zeros_like(pct)))
    y = tf.reduce_sum(tf.where(y_true > 0.001, tf.ones_like(pct), tf.zeros_like(pct)))
    return tf.sqrt(x / y)


class Modes(Enum):
    TRAIN = 0,
    VALID = 1


class DynamicScheduler:
    def __init__(self, opts, verbose=False):
        self.best_loss = np.Inf
        self.fails = 0
        self.lr = (2 ** opts.base_learning_rate) * opts.batch_size
        self.base_lr = self.lr
        self.factor = opts.lr_schedule_plateau_factor
        self.patience = opts.lr_plateau_patience
        self.opts = opts
        self.verbose = verbose

    def schedule(self, loss, step):
        if self.opts.lr_warmup and step <= self.opts.lr_warmup_steps:
            # Scale lr up.
            self.lr = self.base_lr * ((1 / self.opts.batch_size) *
                                      (1 - step / self.opts.lr_warmup_steps) +
                                      step / self.opts.lr_warmup_steps)
            if self.verbose:
                print(f"    Warming up learning rate ({self.base_lr * (1/self.opts.batch_size)} -> {self.base_lr}): {self.lr}")
        else:
            if loss < self.best_loss:
                self.fails = 0
                self.best_loss = loss
            else:
                self.fails += 1
                if self.fails > self.patience:
                    self.lr *= self.factor
                    self.fails = 0
                    if self.verbose:
                        print(f"    Learning stagnated - reduce LR: {self.lr / self.factor} -> {self.lr}")


class ManualScheduler:
    def __init__(self, opts, verbose=False):
        base_lr = (2 ** opts.base_learning_rate) * opts.batch_size
        self.lrs = [base_lr * decay for decay in opts.learning_rate_decay]
        self.lr_drops = [int(i * opts.iterations) for i in opts.learning_rate_schedule]
        self.lr = self.lrs.pop(0)
        self.verbose = verbose

    def schedule(self, loss, epoch):
        if epoch > self.lr_drops[0]:
            del self.lr_drops[0]
            prev_lr = self.lr
            self.lr = self.lrs.pop(0)
            if self.verbose:
                print(f"Dropping learning rate {prev_lr} -> {self.lr}")


class Logger():
    def __init__(self, opts, mode, history_array=[]):
        self.batch_accs = []
        self.batch_times = []
        self.history = history_array
        self.opts = opts
        self.mode = mode
        if self.mode == Modes.TRAIN:
            self.name = "Training"
            self.samples_per_step = opts.batches_per_step * opts.batch_size
        elif self.mode == Modes.VALID:
            self.name = "Validation"
            self.samples_per_step = opts.validation_batches_per_step * opts.validation_batch_size
            self.history_i = 0

    def update(self, i, batch_acc=None, batch_time=None, lr=None, loss=None):
        self.batch_times.append(batch_time)
        if self.opts.multiprocessing and self.mode == Modes.VALID:
            self.history[self.history_i] = batch_acc
            self.history_i += 1
        else:
            self.history.append(batch_acc)

        epoch = float(self.samples_per_step * (i + 1)) / self.opts.training_data._size
        # If mov_mean_window=0 ("all iterations"), then selects entire list
        last_n_batch_times = self.batch_times[-self.opts.mov_mean_window:]

        if not (i+1) % self.opts.steps_per_log or i == 0:
            print(f"{self.name}: step: {i+1:6d}, ",
                  f"epoch: {epoch:6.2f}, ",
                  (f"lr: {lr:6.2g}, " if lr else ""),
                  (f"loss: {loss:6.6f}, " if loss else ""),
                  (f"RMSPE: {batch_acc:6.6f}, " if batch_acc else ""),
                  f"samples/sec: {self.samples_per_step / batch_time:6.2f}, ",
                  f"time: {batch_time:8.6f}, ",
                  f"Moving mean samples/sec ({len(last_n_batch_times)}): {(self.samples_per_step * len(last_n_batch_times)) / np.sum(last_n_batch_times):6.2f}")


class ParallelProcess():
    def __init__(self, target=None, args=None):
        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
                    target=target,
                    args=args + (self.queue,))
        self.process.start()

    def cleanup(self):
        self.queue.put((-1, ""))
        self.queue.close()
        self.queue.join_thread()
        self.process.join()


def is_preprocessed(datafolder):
    # Load train.csv and ensure 'Customers' isn't in the df
    traindf = pd.read_csv(f'{datafolder}/train.csv', low_memory=False)
    return 'Customers' not in traindf


def preprocess_data(datafolder):
    # Read the .csvs
    traindf = pd.read_csv(f'{datafolder}/train.csv', low_memory=False)
    testdf = pd.read_csv(f'{datafolder}/test.csv', low_memory=False)
    storedf = pd.read_csv(f'{datafolder}/store.csv', low_memory=False)
    print("Preprocessing...")

    categorical_cols = [
        'Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'Week', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
        'Promo2SinceYear', 'PromoInterval',
    ]
    continuous_cols = [
        'CompetitionDistance'
    ]
    traindf = traindf.drop(['Customers', 'Open'], axis=1)
    testdf = testdf.drop(['Open', 'Id'], axis=1)

    def modify_df(df):
        initlen = len(df)

        # Expand date into [Year, Month, Day, Week]
        df.Date = pd.to_datetime(df.Date)
        for attr in ['Year', 'Month', 'Day', 'Week']:
            df[attr] = getattr(pd.DatetimeIndex(df.Date), attr.lower())

        # Join store.csv and train/test.csv on Store
        df = pd.merge(df, storedf, on='Store')

        # Fix NaNs in *Since* columns
        defaults = {'CompetitionOpenSinceYear': 1900, 'CompetitionOpenSinceMonth': 1,
                    'Promo2SinceYear': 1900, 'Promo2SinceWeek': 1, 'CompetitionDistance': 0}
        df = df.fillna(defaults)

        # Ensure no rows lost on the join
        assert len(df) == initlen

        # Cast continuous columns to float
        df = df.astype({x: 'float32' for x in continuous_cols})

        # StateHoliday is a special case
        df.StateHoliday = df.StateHoliday.astype(str).replace("0", "d")

        # Move the continuous columns to the end
        for column in continuous_cols:
            cd = df.pop(column)
            df[column] = cd
        return df

    traindf = modify_df(traindf)
    testdf = modify_df(testdf)

    # Filter out 0 sales from train
    traindf = traindf[traindf.Sales > 0]

    # Index categorical columns across both train and test.
    for col in categorical_cols:
        # Get the default type for the column
        default_type = traindf[col].dtype.type() if traindf[col].dtype.type() is not None else ''
        # Fix NaNs with the default value for the column's type.
        traindf[col] = traindf[col].fillna(default_type)
        testdf[col] = testdf[col].fillna(default_type)
        # Get the categories across both train and test
        unique = traindf[col].append(testdf[col]).unique()
        traindf[col] = traindf[col].astype(pd.CategoricalDtype(categories=unique, ordered=True))
        testdf[col] = testdf[col].astype(pd.CategoricalDtype(categories=unique, ordered=True))
        # Assign the categorical columns to their encodings
        traindf[col] = traindf[col].cat.codes
        testdf[col] = testdf[col].cat.codes

    # Move the sales column to the end
    sales = traindf.pop('Sales')
    traindf['Sales'] = sales

    # Make a validation set from the equivalent period of the test set in the training set, a year before.
    t0 = testdf.Date.min() - datetime.timedelta(365)
    t1 = testdf.Date.max() - datetime.timedelta(365)
    val_mask = (traindf.Date > t0) & (traindf.Date <= t1)
    valdf, traindf = traindf[val_mask], traindf[~val_mask]

    # Drop unnecessary columns
    traindf = traindf.drop('Date', axis=1)
    valdf = valdf.drop('Date', axis=1)
    testdf = testdf.drop('Date', axis=1)

    print("Saving to file...")
    traindf.to_csv(f'{datafolder}/train.csv', index=False)
    testdf.to_csv(f'{datafolder}/test.csv', index=False)
    valdf.to_csv(f'{datafolder}/val.csv', index=False)

    print("Done.")
