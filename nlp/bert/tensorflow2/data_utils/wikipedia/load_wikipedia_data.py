# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def get_dataset_files_count(input_files):
    total_samples = 0
    for i, each_input_file in enumerate(input_files, 1):
        total_samples += sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(each_input_file))
        if i % 100 == 0:
            logging.info(f"{i}/{len(input_files)} - total samples = {total_samples} ")
    logging.info(f"The total number of samples in dataset is: {total_samples}")
    return total_samples


def _decode_record(record, name_to_features, data_type=None, debug=False, test=False):
    """
    Decodes a record to a TensorFlow example. In each record, the `input_ids` already have masked tokens (with
    value [MASK]=103). The returned example will have labels masked with 0's for every non [MASK] token.
    """
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        # tf.Example only supports tf.int64, but the IPU only supports tf.int32.
        # So cast all int64 to int32.
        t = example[name]
        if name == "masked_lm_weights" and data_type is not None:
            t = tf.cast(t, dtype=data_type)
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    # Build labels from 'masked_lm_ids' and 'masked_lm_positions'
    masked_lm_ids = example.pop("masked_lm_ids")
    masked_lm_positions = example.pop("masked_lm_positions")
    masked_lm_positions_reshape = tf.reshape(masked_lm_positions, (-1, 1))
    len_seq = len(example["input_ids"])

    mlm_long_labels = tf.scatter_nd(masked_lm_positions_reshape, masked_lm_ids, [len_seq])
    next_sentence_labels = example.pop("next_sentence_labels")

    # Build input, targets tuple and change keys to be compatible with Hugging Face models
    inputs = {
        "input_ids": example.pop("input_ids"),
        "attention_mask": example.pop("input_mask"),
        "token_type_ids": example.pop("segment_ids"),
        "masked_lm_positions": masked_lm_positions,
    }
    labels = (masked_lm_ids, next_sentence_labels)

    if not debug and not test:
        return inputs, labels

    if test:
        labels = (*labels, mlm_long_labels)
        return inputs, labels

    if debug:
        inputs.update({"masked_lm_ids": masked_lm_ids, "next_sentence_labels": next_sentence_labels})
    return inputs, labels


def get_real_dataset(
    micro_batch_size,
    dataset_dir,
    max_seq_length,
    max_predictions_per_seq,
    seed,
    data_type,
    distributed_worker_count,
    distributed_worker_index,
    is_training=True,
    debug=False,
    test=False,
):
    filenames = [
        str(Path(dataset_dir).joinpath(filename))
        for filename in os.listdir(dataset_dir)
        if filename.endswith(".tfrecord")
    ]

    if not filenames:
        raise FileNotFoundError(f"No files found in path {dataset_dir}. Expecting" " files with endings '.tfrecord'.")

    logging.info(f"There are {len(filenames)} input files")

    # Default, the tokens have not been re-arranged.
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }

    buffer_size = 1000

    # For training, we want a lot of parallel reading and shuffling.
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(filenames)
        d = d.repeat()
        d = d.shuffle(buffer_size=len(filenames))

        d = d.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        if distributed_worker_count > 1:
            d = d.shard(num_shards=distributed_worker_count, index=distributed_worker_index)
            d = d.shuffle(buffer_size=buffer_size, seed=seed)
        else:
            d = d.shuffle(buffer_size=buffer_size)
    # For eval, we want no shuffling and parallel reading doesn't matter.
    else:
        d = tf.data.TFRecordDataset(filenames)
        d = d.repeat()

    d = d.map(lambda record: _decode_record(record, name_to_features, data_type, debug, test))
    d = d.batch(batch_size=micro_batch_size, drop_remainder=True)
    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    num_samples = get_dataset_files_count(filenames)

    return d, num_samples


def get_generated_dataset(
    micro_batch_size,
    vocab_size,
    max_seq_length,
    max_predictions_per_seq,
    distributed_worker_count,
    distributed_worker_index,
):
    logging.info("Generating artificial data for pretraining.")
    generated_dataset_size = 100000
    _input_ids = np.random.randint(vocab_size, size=(generated_dataset_size, max_seq_length)).astype(np.int32)
    _input_mask = np.random.randint(2, size=(generated_dataset_size, max_seq_length)).astype(np.int32)
    _segment_ids = np.random.randint(2, size=(generated_dataset_size, max_seq_length)).astype(np.int32)
    _masked_lm_positions = np.random.randint(
        max_seq_length, size=(generated_dataset_size, max_predictions_per_seq)
    ).astype(np.int32)
    _masked_lm_ids = np.random.randint(vocab_size, size=(generated_dataset_size, max_predictions_per_seq)).astype(
        np.int32
    )
    _masked_lm_weights = np.random.randint(2, size=(generated_dataset_size, max_predictions_per_seq)).astype(np.float16)
    _next_sentence_labels = np.random.randint(2, size=(generated_dataset_size, 1)).astype(np.int32)
    # Build input, targets tuple and change keys to be compatible with Hugging Face models
    data = {
        "input_ids": _input_ids,
        "attention_mask": _input_mask,
        "token_type_ids": _segment_ids,
        "masked_lm_positions": _masked_lm_positions,
    }
    labels = (_masked_lm_ids, _next_sentence_labels)
    inputs = (data, labels)
    d = tf.data.Dataset.from_tensor_slices(inputs)
    if distributed_worker_count > 1:
        d = d.shard(num_shards=distributed_worker_count, index=distributed_worker_index)
    d = d.repeat()
    d = d.shuffle(buffer_size=generated_dataset_size)
    d = d.batch(batch_size=micro_batch_size, drop_remainder=True)
    d = d.prefetch(tf.data.experimental.AUTOTUNE)

    return d, generated_dataset_size


def get_pretraining_dataset(
    micro_batch_size,
    dataset_dir,
    max_seq_length,
    max_predictions_per_seq,
    vocab_size,
    seed,
    data_type,
    distributed_worker_count,
    distributed_worker_index,
    generated_dataset=False,
    is_training=True,
    debug=False,
    test=False,
):
    if generated_dataset:
        return get_generated_dataset(
            micro_batch_size,
            vocab_size,
            max_seq_length,
            max_predictions_per_seq,
            distributed_worker_count,
            distributed_worker_index,
        )

    else:
        return get_real_dataset(
            micro_batch_size,
            dataset_dir,
            max_seq_length,
            max_predictions_per_seq,
            seed,
            data_type,
            distributed_worker_count,
            distributed_worker_index,
            is_training=is_training,
            debug=debug,
            test=test,
        )
