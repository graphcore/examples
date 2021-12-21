# coding=utf-8
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

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib.data import map_and_batch, parallel_interleave


def _decode_record(record, name_to_features, data_type=None):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if name == 'masked_lm_weights' and data_type is not None:
            t = tf.cast(
                t, dtype=data_type)
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

        # For compatibility with older datasets
        if name == 'masked_labels':
            example['masked_lm_ids'] = example.pop('masked_labels')

    return example


def synthetic_pretraining_dataset(opts):
    """Returns dataset filled with random data."""
    vocab_size = opts['vocab_size']
    micro_batch_size = opts['micro_batch_size']
    seq_length = opts['seq_length']
    max_predictions_per_seq = opts['max_predictions_per_seq']
    generated_dataset_size = 1000

    if opts['static_mask']:
        _input_ids = np.random.randint(vocab_size, size=(generated_dataset_size, seq_length)).astype(np.int32)
        _input_position = np.random.randint(seq_length, size=(generated_dataset_size, seq_length)).astype(np.int32)
        _token_type_ids = np.random.randint(4, size=(generated_dataset_size, seq_length)).astype(np.int32)
        _masked_lm_positions = np.random.randint(seq_length, size=(generated_dataset_size, max_predictions_per_seq)).astype(np.int32)
        _masked_lm_ids = np.random.randint(vocab_size, size=(generated_dataset_size, max_predictions_per_seq)).astype(np.int32)
        _masked_lm_weights = np.random.randint(2, size=(generated_dataset_size, max_predictions_per_seq)).astype(np.float16)
        _next_sentence_labels = np.random.randint(1000, size=(generated_dataset_size, 1)).astype(np.int32)
        _mask_padding_index = np.random.randint(20, size=(generated_dataset_size, 1)).astype(np.int32)
        _seq_padding_index = np.random.randint(128, size=(generated_dataset_size, 1)).astype(np.int32)
        name_dict = {
            'input_ids': _input_ids,
            'input_position': _input_position,
            "segment_ids": _token_type_ids,
            "mask_padding_index": _mask_padding_index,
            "seq_padding_index": _seq_padding_index,
            "masked_lm_ids": _masked_lm_ids,
            "masked_lm_weights": _masked_lm_weights,
            "next_sentence_labels": _next_sentence_labels
        }
    else:
        _input_ids = np.random.randint(vocab_size, size=(generated_dataset_size, seq_length)).astype(np.int32)
        _input_mask = np.random.randint(2, size=(generated_dataset_size, seq_length)).astype(np.int32)
        _segment_ids = np.random.randint(2, size=(generated_dataset_size, seq_length)).astype(np.int32)
        _masked_lm_positions = np.random.randint(seq_length, size=(generated_dataset_size, max_predictions_per_seq)).astype(np.int32)
        _masked_lm_ids = np.random.randint(vocab_size, size=(generated_dataset_size, max_predictions_per_seq)).astype(np.int32)
        _masked_lm_weights = np.random.randint(2, size=(generated_dataset_size, max_predictions_per_seq)).astype(np.float16)
        _next_sentence_labels = np.random.randint(2, size=(generated_dataset_size, 1)).astype(np.int32)
        name_dict = {
            "input_ids": _input_ids,
            "input_mask": _input_mask,
            "segment_ids": _segment_ids,
            "masked_lm_positions": _masked_lm_positions,
            "masked_lm_ids": _masked_lm_ids,
            "masked_lm_weights": _masked_lm_weights,
            "next_sentence_labels": _next_sentence_labels
        }
    d = tf.data.Dataset.from_tensor_slices(name_dict)
    if opts['distributed_worker_count'] > 1:
        d = d.shard(num_shards=opts['distributed_worker_count'], index=opts['distributed_worker_index'])
    dataset = d.cache()
    dataset = dataset.repeat()
    return dataset.batch(batch_size=micro_batch_size, drop_remainder=True)


def get_pretraining_dataset(opts, data_type, is_training=True, num_cpu_threads=4, use_static_mask=False):
    if is_training:
        input_file = opts['train_file']
    else:
        input_file = opts['test_file']
    micro_batch_size = opts['micro_batch_size']
    max_seq_length = opts['seq_length']
    max_predictions_per_seq = opts['max_predictions_per_seq']

    input_files = []
    for input_pattern in input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    if use_static_mask:
        # The masked tokens have been re-arranaged to always be at the first
        # 'max_predictions_per_seq' positions.
        name_to_features = {
            "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_position":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "mask_padding_index":
            tf.FixedLenFeature([1], tf.int64),
            "seq_padding_index":
            tf.FixedLenFeature([1], tf.int64),
            "masked_labels":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
        }
    else:
        # Default, the tokens have not been re-arranged.
        name_to_features = {
            "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
        }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.repeat()

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(parallel_interleave(tf.data.TFRecordDataset, sloppy=is_training, cycle_length=cycle_length))

        # `buffer_size` should be set big enough to keep data shuffle sufficiently.
        if opts['distributed_worker_count'] > 1:
            d = d.shard(num_shards=opts['distributed_worker_count'], index=opts['distributed_worker_index'])
            d = d.shuffle(buffer_size=1000, seed=opts['seed'])
        else:
            d = d.shuffle(buffer_size=1000)
    else:
        d = tf.data.TFRecordDataset(input_files)
        d = d.repeat()

    d = d.apply(map_and_batch(
            lambda record: _decode_record(record, name_to_features, data_type),
            batch_size=micro_batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d
