# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import logging

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from transformers import AutoTokenizer



def get_encodings_from_example(example, tokenizer, task_keys, max_seq_length, return_offsets_mapping=False):
    label = tf.cast(example["label"], dtype=tf.int32)
    sentence1 = example[task_keys[0]].numpy().decode("utf-8")
    if task_keys[1] is not None:
        sentence2 = example[task_keys[1]].numpy().decode("utf-8")
    else:
        sentence2 = None
    idx = example["idx"].numpy()
    if max_seq_length is not None:
        max_length = max_seq_length
    else:
        max_length = tokenizer.model_max_length
    encodings = tokenizer.encode_plus(
        sentence1,
        sentence2,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=return_offsets_mapping
    )
    return encodings, label, idx


def convert_example_to_tf_features(example, tokenizer, task_keys, max_seq_length):
    encodings, label, idx = get_encodings_from_example(example, tokenizer, task_keys, max_seq_length)
    return encodings, {'labels': label}


def convert_to_features(dataset, tokenizer, convert_example_func, task_keys, max_seq_length):
    labels = {}
    features = {}
    for item in tqdm(dataset):
        feature, label = convert_example_func(
            item, tokenizer, task_keys, max_seq_length)
        if feature is not None and label is not None:
            # This catches the GLUE tasks with 1 or 2 sentences.
            for k, v in feature.items():
                features.setdefault(k, []).append(tf.constant(v))
            for k, v in label.items():
                labels.setdefault(k, []).append(tf.constant(v))
    return features, labels


def get_glue_data(task, micro_batch_size, cache_dir, max_seq_length=None, generated_dataset=False):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    assert task in task_to_keys, f"The task {task} is not supported.\n" + \
        f"Supported task are {task_to_keys}"
    if generated_dataset:
        raw_datasets = get_generated_dataset()
    else:
        raw_datasets = tfds.load(f"glue/{task}", data_dir=cache_dir)
    num_train_samples = raw_datasets['train'].cardinality()
    num_eval_samples = raw_datasets['validation'].cardinality()
    num_test_samples = raw_datasets['test'].cardinality()

    # The tokenizer used for fine-tuning is the same as in pre-training
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    logging.info('Computing features for the training dataset')
    train_features, train_label = convert_to_features(
        raw_datasets['train'],
        tokenizer,
        convert_example_to_tf_features,
        task_to_keys[task],
        max_seq_length
    )
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, train_label))
    train_dataset = train_dataset.shuffle(num_train_samples)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(micro_batch_size, drop_remainder=True)

    logging.info("Computing features for the validation dataset")
    eval_features, eval_labels = convert_to_features(
        raw_datasets['validation'],
        tokenizer,
        convert_example_to_tf_features,
        task_to_keys[task],
        max_seq_length
    )
    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_features, eval_labels)).batch(
        micro_batch_size, drop_remainder=True)

    logging.info("Computing features for the test dataset")
    test_features, test_labels = convert_to_features(
        raw_datasets['test'],
        tokenizer,
        convert_example_to_tf_features,
        task_to_keys[task],
        max_seq_length
    )
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(
        micro_batch_size, drop_remainder=True)

    return train_dataset, eval_dataset, test_dataset, num_train_samples, num_eval_samples, num_test_samples, raw_datasets


def get_generated_dataset():
    """Generate fake GLUE examples for tests and debugging."""
    num_generated = 12000
    dummy_text = {"idx": [tf.constant(str(dummy_id)) for dummy_id in range(0, num_generated)],
                  "sentence1": [tf.constant("Dummy Graphcore Text")] * num_generated,
                  "sentence2": [tf.constant("Graphcore Dummy Text")] * num_generated,
                  "label": [tf.constant(0) for dummy_id in range(0, num_generated)]}
    generated_data = {"train": tf.data.Dataset.from_tensor_slices(dummy_text),
                      "validation": tf.data.Dataset.from_tensor_slices(dummy_text),
                      "test": tf.data.Dataset.from_tensor_slices(dummy_text)}
    return generated_data
