# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""CosmoFlow dataset specification"""

import os
import logging
import glob
from functools import partial

import numpy as np
import tensorflow as tf


def _parse_data(sample_proto, shape, apply_log=False):
    """Parse the data out of the TFRecord proto buf.

    This pipeline could be sped up considerably by moving the cast and log
    transform onto the GPU, in the model (e.g. in a keras Lambda layer).
    """

    # Parse the serialized features
    feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string),
                        y=tf.io.FixedLenFeature([4], tf.float32))
    parsed_example = tf.io.parse_single_example(
        sample_proto, features=feature_spec)

    # Decode the bytes data, convert to float
    x = tf.decode_raw(parsed_example['x'], tf.int16)
    x = tf.cast(tf.reshape(x, shape), tf.float32)
    y = parsed_example['y']

    # Data normalization/scaling
    if apply_log:
        # Take logarithm of the data spectrum
        x = tf.math.log(x + tf.constant(1.))
    else:
        # Traditional mean normalization
        x /= (tf.reduce_sum(x) / np.prod(shape))

    return x, y


def construct_dataset(file_dir, n_samples, batch_size, n_epochs,
                      sample_shape, samples_per_file=1, n_file_sets=1,
                      shard=0, n_shards=1, apply_log=False,
                      shuffle=False, shuffle_buffer_size=0,
                      prefetch=4):
    """This function takes a folder with files and builds the TF dataset.

    It ensures that the requested sample counts are divisible by files,
    local-disks, worker shards, and mini-batches.
    """

    if n_samples == 0:
        return None, 0

    # Ensure samples divide evenly into files * local-disks * worker-shards * batches
    n_divs = samples_per_file * n_file_sets * n_shards * batch_size
    if (n_samples % n_divs) != 0:
        logging.error('Number of samples (%i) not divisible by %i '
                      'samples_per_file * n_file_sets * n_shards * batch_size',
                      n_train, n_divs)
        raise Exception('Invalid sample counts')

    # Number of files and steps
    n_files = n_samples // (samples_per_file * n_file_sets)
    n_steps = n_samples // (n_file_sets * n_shards * batch_size)

    # Find the files
    filenames = sorted(glob.glob(os.path.join(file_dir, '*.tfrecord')))
    if len(filenames) < n_files:
        logging.error('Requested %i files, but only found %i', n_files, len(filenames))
        raise Exception('Invalid file counts')

    # Define the dataset from the list of sharded, shuffled files
    data = tf.data.Dataset.from_tensor_slices(filenames)
    data = data.shard(num_shards=n_shards, index=shard)
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)

    # Parse TFRecords
    parse_data = partial(_parse_data, shape=sample_shape, apply_log=apply_log)
    data = data.apply(tf.data.TFRecordDataset).map(parse_data, num_parallel_calls=4)

    # Localized sample shuffling (note: imperfect global shuffling).
    # Use if samples_per_file is greater than 1.
    if shuffle and shuffle_buffer_size > 0:
        data = data.shuffle(shuffle_buffer_size)

    # Construct batches
    data = data.repeat(n_epochs)
    data = data.batch(batch_size, drop_remainder=True)

    # Prefetch to device
    return data.prefetch(prefetch), n_steps


def get_datasets(data_dir, sample_shape, n_train, n_valid,
                 batch_size, n_epochs, dist=None, samples_per_file=1,
                 shuffle_train=True, shuffle_valid=False,
                 shard=True, staged_files=False,
                 prefetch=4, apply_log=False):
    """Prepare TF datasets for training and validation.

    This function figures out how to split files according to local filesystems
    (if pre-staging) and worker shards (if sharding).

    Returns: A dict of the two datasets and step counts per epoch.
    """

    # Determine number of staged file sets and worker shards
    n_file_sets = (dist.size // dist.local_size) if staged_files else 1
    if shard and staged_files:
        shard, n_shards = dist.local_rank, dist.local_size
    elif shard and not staged_files:
        shard, n_shards = dist.rank, dist.size
    else:
        shard, n_shards = 0, 1

    # Construct the training and validation datasets
    dataset_args = dict(batch_size=batch_size, n_epochs=n_epochs,
                        sample_shape=sample_shape, samples_per_file=samples_per_file,
                        n_file_sets=n_file_sets, shard=shard, n_shards=n_shards,
                        apply_log=apply_log, prefetch=prefetch)
    train_dataset, n_train_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'train'),
        n_samples=n_train, shuffle=shuffle_train, **dataset_args)
    valid_dataset, n_valid_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'validation'),
        n_samples=n_valid, shuffle=shuffle_valid, **dataset_args)

    if shard == 0:
        if staged_files:
            logging.info('Using %i locally-staged file sets', n_file_sets)
        logging.info('Splitting data into %i worker shards', n_shards)
        n_train_worker = n_train // (samples_per_file * n_file_sets * n_shards)
        n_valid_worker = n_valid // (samples_per_file * n_file_sets * n_shards)
        logging.info('Each worker reading %i training samples and %i validation samples',
                     n_train_worker, n_valid_worker)

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train_steps=n_train_steps, n_valid_steps=n_valid_steps)
