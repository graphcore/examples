# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import numpy as np
import os

# Data file settings
USER_PART = 0  # user id goes first in the data file
ITEM_PART = 1  # item id goes second in the data file
RATING_PART = 2  # rating goes third in the data file
DELIMITER = '\t'

# Dataset generation parameters
SHUFFLE_BUFFER = 10000


class AutoencoderData:
    def __init__(self, data_file_name, training_data=None):
        self.data_file_name = data_file_name

        # When doing validation, we use training data to feed into the model and generate predicted ratings.
        # The predictions are then compared to the ground truth in validation
        # data.
        if training_data is None:
            self.user_xlat_map, self.item_xlat_map = {}, {}
            do_build_maps = True
            user_index, item_index = 0, 0
        else:
            self.training_dict = training_data.ratings_dict
            self.user_xlat_map, self.item_xlat_map = training_data.user_xlat_map, training_data.item_xlat_map
            do_build_maps = False

        self.ratings_dict = {}
        observed_rating_events = []
        if self.data_file_name:
            observed_rating_events = np.loadtxt(self.data_file_name, delimiter=DELIMITER)
        else:
            # If no data file is provided, generate random data
            num_events = 13000000 if training_data is None else 1000000
            observed_rating_events = self.generate_random_ratings(num_events=num_events)

        for i in range(observed_rating_events.shape[0]):
            user_id_external, item_id_external, rating = tuple(observed_rating_events[i])
            # Build dense maps to keep only users or items that have
            # training data
            if do_build_maps:
                if user_id_external not in self.user_xlat_map:
                    self.user_xlat_map[user_id_external] = user_index
                    user_index += 1
                if item_id_external not in self.item_xlat_map:
                    self.item_xlat_map[item_id_external] = item_index
                    item_index += 1

            # Mapped user and item ids
            user_id_internal = self.user_xlat_map[user_id_external]
            item_id_internal = self.item_xlat_map[item_id_external]
            if user_id_internal not in self.ratings_dict:
                self.ratings_dict[user_id_internal] = []
            self.ratings_dict[user_id_internal].append(
                (item_id_internal, rating))

        self.user_id_list = list(self.ratings_dict.keys())
        self._input_size = len(self.item_xlat_map)

    # Generate random ratings (used when real data is not available)

    def generate_random_ratings(self, num_events, num_users = 480000, num_items = 18000):
        user_id_external = np.random.randint(num_users, size=[num_events])
        item_id_external = np.random.randint(num_items, size=[num_events])
        rating = np.random.randint(5, size=[num_events]) + 1
        return np.stack([user_id_external, item_id_external, rating], axis=1)

    # Generate one data sample during training

    def generate_sample(self):
        index = 0
        while index < len(self.user_id_list):
            sample = np.zeros(self._input_size, dtype=np.int8)
            for (item_number, rating) in self.ratings_dict[self.user_id_list[index]]:
                sample[item_number] = rating
            index += 1
            yield sample

    # Generate a pair of observed and ground truth data samples during
    # evaluation

    def generate_pair(self):
        index = 0
        while index < len(self.user_id_list):
            # sample ground truth
            sample_gt = np.zeros(self._input_size, dtype=np.int8)
            for (item_number, rating) in self.ratings_dict[self.user_id_list[index]]:
                sample_gt[item_number] = rating
            # sample observed
            sample_observed = np.zeros(self._input_size, dtype=np.int8)
            for (item_number, rating) in self.training_dict[self.user_id_list[index]]:
                sample_observed[item_number] = rating
            index += 1
            yield np.concatenate((sample_observed, sample_gt), axis=None)

    @property
    def size(self):
        return len(self.user_id_list)

    @property
    def input_size(self):
        return self._input_size

    # Create a tf Dataset with buffering, stats, etc.

    def get_dataset(self, opts, is_training):
        micro_batch_size = opts.micro_batch_size if is_training else opts.validation_micro_batch_size
        dtypes = opts.precision.split('.')
        datatype = tf.float16 if dtypes[0] == '16' else tf.float32

        # Create a tf Dataset using a binary file or from generator
        if is_training:
            # We use a binary file to store the preprocessed dataset.
            # This way the throughput is more stable.
            def generate_entire_dataset_in_numpy():
                index = 0
                samples = np.zeros((self.size, self._input_size), dtype=np.int8)
                while index < self.size:
                    for (item_number, rating) in self.ratings_dict[self.user_id_list[index]]:
                        samples[index, item_number] = rating
                    index += 1
                return samples

            if opts.training_data_file:
                filename = os.path.splitext(opts.training_data_file)[0]+'.bin'
            else:
                filename = 'generated_random_training_data.bin'
            if not os.path.exists(filename):
                print('Writing to {}'.format(filename))
                generate_entire_dataset_in_numpy().astype('int8').tofile(filename)
            dataset = tf.data.FixedLengthRecordDataset([filename], record_bytes=self._input_size)
        else:
            # For validation, create Dataset from generator
            dataset = tf.data.Dataset.from_generator(self.generate_pair,
                                                     output_types=datatype,
                                                     output_shapes=tf.TensorShape([2 * self._input_size]))
            dataset = dataset.cache(opts.logs_path + '/$$validation_data_cache$$')

        # Shuffle and/or repeat
        if is_training:
            dataset = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(SHUFFLE_BUFFER))
        else:
            dataset = dataset.repeat()

        if is_training:
            # Decode binary records and batch the data
            def decode_binary(raw_record):
                record_vector = tf.decode_raw(raw_record, tf.int8)
                reshaped = tf.reshape(record_vector, [self._input_size])
                return tf.cast(reshaped, datatype)

            dataset = dataset.apply(
                            tf.data.experimental.map_and_batch(
                                decode_binary,
                                micro_batch_size,
                                num_parallel_calls=opts.pipeline_num_parallel,
                                drop_remainder=True
                            )
                        )
        else:
            # Batch
            dataset = dataset.batch(
                micro_batch_size,
                drop_remainder=True
            )

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

        iterator = dataset.make_initializable_iterator()
        placeholders = {}

        if is_training:
            placeholders['learning_rate'] = tf.placeholder(datatype, shape=[])

        return dataset, iterator, placeholders
