# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""
Random dummy dataset specification.
"""

# System
import math

# Externals
import tensorflow as tf


def construct_dataset(sample_shape, target_shape, micro_batch_size=1, n_samples=32):

    datatype = tf.float32

    input_data = tf.random.uniform(
        sample_shape,
        dtype=datatype,
        name='synthetic_inputs')

    labels = tf.random.uniform(
        target_shape,
        dtype=datatype,
        name='synthetic_labels')

    data = tf.data.Dataset.from_tensors((input_data, labels))

    data = data.repeat(n_samples)
    data = data.batch(batch_size=micro_batch_size, drop_remainder=True)
    data = data.cache()
    data = data.repeat()
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data


def get_datasets(sample_shape, target_shape, micro_batch_size,
                 n_train, n_valid, n_epochs=None, shard=False,
                 rank=0, n_ranks=1):
    train_dataset = construct_dataset(sample_shape, target_shape, micro_batch_size=micro_batch_size)
    valid_dataset = None
    if n_valid > 0:
        valid_dataset = construct_dataset(sample_shape, target_shape, micro_batch_size=micro_batch_size)
    n_train_steps = n_train // micro_batch_size
    n_valid_steps = n_valid // micro_batch_size
    if shard:
        n_train_steps = n_train_steps // n_ranks
        n_valid_steps = n_valid_steps // n_ranks

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
