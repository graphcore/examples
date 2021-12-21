# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path

import tensorflow as tf


def get_dataset_files_count(input_files):
    total_samples = 0
    for i, each_input_file in enumerate(input_files, 1):
        total_samples += sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(each_input_file))
        if i % 100 == 0:
            print(f"{i}/{len(input_files)} - total samples = {total_samples} ")
    print(f"The total number of samples in dataset is: {total_samples}")
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
        if name == 'masked_lm_weights' and data_type is not None:
            t = tf.cast(t, dtype=data_type)
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    # Build labels from 'masked_lm_ids' and 'masked_lm_positions'
    masked_lm_ids = example.pop('masked_lm_ids')
    masked_lm_positions = example.pop('masked_lm_positions')
    masked_lm_positions_reshape = tf.reshape(masked_lm_positions, (-1, 1))
    len_seq = len(example['input_ids'])

    mlm_long_labels = tf.scatter_nd(masked_lm_positions_reshape, masked_lm_ids, [len_seq])
    next_sentence_labels = example.pop('next_sentence_labels')

    # Build input, targets tuple and change keys to be compatible with Hugging Face models
    inputs = {
                'input_ids': example.pop('input_ids'),
                'attention_mask': example.pop('input_mask'),
                'token_type_ids': example.pop('segment_ids'),
                'masked_lm_positions': masked_lm_positions
            }
    labels = (masked_lm_ids, next_sentence_labels)

    if not debug and not test:
        return inputs, labels

    if test:
        labels = (*labels, mlm_long_labels)
        return inputs, labels

    if debug:
        inputs.update(
            {
                'masked_lm_ids': masked_lm_ids,
                'next_sentence_labels': next_sentence_labels
            }
        )
    return inputs, labels


def get_pretraining_dataset(micro_batch_size,
                            dataset_dir,
                            max_seq_length,
                            max_predictions_per_seq,
                            distributed_worker_count,
                            seed,
                            data_type,
                            is_training=True,
                            num_cpu_threads=4,
                            debug=False,
                            test=False):

    filenames = [str(Path(dataset_dir).joinpath(filename))
                 for filename in os.listdir(dataset_dir)
                 if filename.endswith('.tfrecord')]

    if not filenames:
        raise FileNotFoundError(f"No files found in path {dataset_dir}. Expecting"
                                " files with endings '.tfrecord'.")

    print(f"There are {len(filenames)} input files")

    # Default, the tokens have not been re-arranged.
    name_to_features = {
        "input_ids":
        tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
        tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
        tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
        tf.io.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        # d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = tf.data.Dataset.from_tensor_slices(filenames)
        d = d.repeat()

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(filenames))

        # Not deterministic interleaving adds even more randomness to the training pipeline.
        deterministic = not is_training
        d = d.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=cycle_length, deterministic=deterministic)

        # `buffer_size` should be set big enough to keep data shuffle sufficiently.
        if distributed_worker_count > 1:
            # TODO: Do we need this case?
            d = d.shard(num_shards=distributed_worker_count, index=distributed_worker_count)
            d = d.shuffle(buffer_size=10000, seed=seed)
        else:
            d = d.shuffle(buffer_size=10000)
    else:
        d = tf.data.TFRecordDataset(filenames)
        d = d.repeat()

    d = d.map(lambda record: _decode_record(record, name_to_features, data_type, debug, test))
    d = d.batch(batch_size=micro_batch_size, drop_remainder=True)
    return d, filenames
