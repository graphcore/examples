# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf

import util

# Data pipeline parameters
SHUFFLE_BUFFER = 10000


class MLPData:
    def __init__(self, opts, data_path):
        # Define data constants - these must be hardcoded.
        # Warning: data requires categorical columns in order, then continuous columns in order, then sales column
        self.CATEGORICAL_COLS = [
            'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Day', 'Week', 'StoreType', 'Assortment',
            'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'
        ]
        self.CONTINUOUS_COLS = [
            'CompetitionDistance'
        ]
        self.SYNTHETIC_NUM_ROWS = 807691
        self.VOCAB_LENS = [1115, 7, 2, 4, 2, 3, 12, 31, 52, 4, 3, 12, 23, 2, 24, 8, 4]
        self.NUM_CATEGORICAL = len(self.CATEGORICAL_COLS)
        self.NUM_CONTINUOUS = len(self.CONTINUOUS_COLS)
        self.NUM_COLS = self.NUM_CATEGORICAL + self.NUM_CONTINUOUS + 1  # Adding sales column

        dtype = getattr(np, opts.dtypes[0])
        self.data_path = data_path

        # If using synthetic data, data_path = None. Generate placeholder data instead.
        if not data_path:
            print(" Using synthetic data")
            self._size = self.SYNTHETIC_NUM_ROWS
            self._data = np.ones([self._size, self.NUM_COLS], dtype=dtype)
        else:
            # Load CSV into a numpy array.
            # First load in as FP32, as some columns may contain values > max(FP16). Then log such columns, then cast to 16.
            self._data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32, skip_header=1)

            # Log the 'Sales' column. Sales column is at the end
            self._data[:, -1] = np.log(self._data[:, -1])

            # Log the 'CompetitionDistance' column otherwise moving_variance is too large for 16.16 to represent in BNorm update parameters
            # Add 1 to avoid log(0) and map 0 -> log(0+1) -> 0
            self._data[:, -2] = np.log(self._data[:, -2] + 1)

            # Cast to dtype
            self._data = self._data.astype(dtype)

            # Store dataset size
            self._size = self._data.shape[0]


        # Determine log max Sales number
        self._log_max_sales = np.amax(self._data[:, -1])

    @property
    def size(self):
        return self._size

    def get_dataset(self, opts, mode):
        ''' Create a tf Dataset with buffering, stats, etc. '''
        dtype = getattr(np, opts.dtypes[0])
        batch_size = opts.batch_size if mode == util.Modes.TRAIN else opts.validation_batch_size

        # Create tf Dataset from the numpy array
        dataset = tf.data.Dataset.from_tensor_slices((self._data))
        # Repeat, and shuffle if we're training
        if mode == util.Modes.TRAIN:
            dataset = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(SHUFFLE_BUFFER))
        else:
            dataset = dataset.repeat()
        # Batch the data
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # Prefetch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Pipeline stats
        stats_aggregator = tf.data.experimental.StatsAggregator()
        dataset = dataset.apply(
            tf.data.experimental.latency_stats("latency_stats"))
        tf.add_to_collection(
            tf.GraphKeys.SUMMARIES,
            stats_aggregator.get_summary())
        options = tf.data.Options()
        options.experimental_stats.aggregator = stats_aggregator
        dataset = dataset.with_options(options)

        placeholders = {}

        if mode == util.Modes.TRAIN:
            placeholders['learning_rate'] = tf.placeholder(dtype, shape=[])

        return dataset, placeholders
