# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
from transformers import AutoTokenizer
from tqdm import tqdm


# Functions for data loading
# ==========================


def get_correct_alignement(context, answer, start_idx):
    end_idx = start_idx + len(answer)
    if context[start_idx:end_idx] == answer:
        return start_idx, end_idx  # When the answer label position is good
    elif context[start_idx - 1 : end_idx - 1] == answer:
        return start_idx - 1, end_idx - 1  # When the answer label is off by one character
    elif context[start_idx - 2 : end_idx - 2] == answer:
        return start_idx - 2, end_idx - 2  # When the answer label is off by two character
    else:
        raise ValueError()


def get_encodings_and_start_end_positions_from_example(
    example, tokenizer, max_seq_length, return_offsets_mapping=False
):
    context = example["context"].numpy().decode("utf-8")
    question = example["question"].numpy().decode("utf-8")
    if max_seq_length is not None:
        max_length = max_seq_length
    else:
        max_length = tokenizer.model_max_length
    encodings = tokenizer.encode_plus(
        context,
        question,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=return_offsets_mapping,
    )
    start_positions, end_positions = [], []

    first_answer = example["answers"]["text"][0].numpy().decode("utf-8") if len(example["answers"]["text"]) > 0 else ""
    first_answer_start = example["answers"]["answer_start"][0].numpy()
    start_idx, end_idx = get_correct_alignement(context, first_answer, first_answer_start)
    start = encodings.char_to_token(0, start_idx)
    end = encodings.char_to_token(0, end_idx - 1) if end_idx > 0 else 0

    if start is None or end is None:
        return None, None, None

    start_positions.append(start)
    end_positions.append(end)
    return encodings, start_positions, end_positions


def convert_train_example_to_tf_features(example, tokenizer, max_seq_length):
    encodings, start_positions, end_positions = get_encodings_and_start_end_positions_from_example(
        example, tokenizer, max_seq_length
    )
    return encodings, {"start_positions": start_positions, "end_positions": end_positions}


def convert_validation_example_to_tf_features(example, tokenizer, max_seq_length):
    encodings, start_positions, end_positions = get_encodings_and_start_end_positions_from_example(
        example, tokenizer, max_seq_length, return_offsets_mapping=True
    )
    example_id = example["id"].numpy().decode("utf-8")
    return encodings, {"start_positions": start_positions, "end_positions": end_positions, "id": example_id}


def convert_to_features(dataset, tokenizer, max_seq_length, convert_example_func):
    features = {}
    labels = {}
    for item in tqdm(dataset):
        feature, label = convert_example_func(item, tokenizer, max_seq_length)
        if feature is not None and label is not None:
            for k, v in feature.items():
                features.setdefault(k, []).append(tf.constant(v))
            for k, v in label.items():
                labels.setdefault(k, []).append(tf.constant(v))
    return features, labels


def get_squad_data(micro_batch_size, cache_dir, generated_dataset=False, max_seq_length=None):
    if generated_dataset:
        logging.info("Generating artificial data for SQuAD.")
        raw_datasets = generated_squad_dataset()
        num_train_samples = raw_datasets["train"].cardinality()
        num_eval_samples = raw_datasets["validation"].cardinality()
    else:
        raw_datasets = tfds.load("squad", data_dir=cache_dir)
        num_train_samples = raw_datasets["train"].cardinality()
        num_eval_samples = raw_datasets["validation"].cardinality()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    logging.info("Computing features for the training dataset")
    train_features, train_labels = convert_to_features(
        raw_datasets["train"], tokenizer, max_seq_length, convert_train_example_to_tf_features
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_dataset = train_dataset.shuffle(num_train_samples)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(micro_batch_size, drop_remainder=True)

    logging.info("Computing features for the validation dataset")
    eval_features, eval_labels = convert_to_features(
        raw_datasets["validation"], tokenizer, max_seq_length, convert_validation_example_to_tf_features
    )
    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_features, eval_labels)).batch(
        micro_batch_size, drop_remainder=True
    )

    return train_dataset, eval_dataset, num_train_samples, num_eval_samples, raw_datasets


#  Functions for data validation
#  =============================


def get_prediction_dataset(dataset, max_batches):
    pred_dataset = dataset.take(max_batches)
    pred_dataset = pred_dataset.map(
        lambda inputs, labels: (
            {key: val for key, val in inputs.items() if key != "offset_mapping"},
            {key: val for key, val in labels.items() if key != "id"},
        )
    )
    return pred_dataset


def format_raw_data_for_metric(dataset):
    references = [
        {
            "id": ex["id"].numpy().decode(),
            "answers": {
                "answer_start": list(ex["answers"]["answer_start"].numpy()),
                "text": [item.decode() for item in ex["answers"]["text"].numpy()],
            },
        }
        for ex in dataset
    ]
    return references


def generated_squad_dataset():
    """Generate fake SQuAD examples for tests and debugging."""
    num_generated = 12000
    dummy_text = {
        "id": [tf.constant(str(dummy_id)) for dummy_id in range(0, num_generated)],
        "title": [tf.constant("Dummy Graphcore Text")] * num_generated,
        "context": [tf.constant("SQUAD is a question answer dataset. There are loads of questions.")] * num_generated,
        "question": [tf.constant("What is SQUAD?")] * num_generated,
        "answers": {
            "text": [[tf.constant("dataset")]] * num_generated,
            "answer_start": [[tf.constant(27)]] * num_generated,
        },
    }
    generated_data = {
        "train": tf.data.Dataset.from_tensor_slices(dummy_text),
        "validation": tf.data.Dataset.from_tensor_slices(dummy_text),
    }
    return generated_data
