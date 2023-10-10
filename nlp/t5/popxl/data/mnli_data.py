# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def form_training_prompts(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]
    class_label = ["entailment", "neutral", "contradiction"][example["label"]]

    example["text"] = f"mnli hypothesis: {hypothesis} premise: {premise}"
    example["target"] = f"{class_label}"
    return example


def form_validation_prompts(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]

    example["text"] = f"mnli hypothesis: {hypothesis} premise: {premise}"
    return example


def prepare_validation_features(tokenizer):
    def func(dataset):
        tokenized = tokenizer(dataset["text"], padding="max_length", return_tensors="np")
        tokenized.update(label=dataset["label"])
        return tokenized

    return func


def extract_class_label(s: str) -> str:
    """Extract a class label from the generated text."""
    s = s.strip()
    # no need if decoded using skip special tokens
    s = s.replace("</s>", "")
    if s in ["entailment", "neutral", "contradiction"]:
        return s
    else:
        return "unknown"


def postprocess_mnli_predictions(generated_sentences):
    labels_to_ids = {"entailment": 0, "neutral": 1, "contradiction": 2, "unknown": -1}
    predictions = []
    for s in generated_sentences:
        answer = extract_class_label(s)
        predictions.append(labels_to_ids[answer])
    return predictions


def tokenizes_text(tokenizer):
    def func(dataset):
        # Tokenise the input text and the target text
        tokenized = tokenizer(dataset["text"], padding="max_length", truncation=True, return_tensors="np")
        target_tokenized = tokenizer(dataset["target"], padding="max_length", truncation=True, return_tensors="np")
        # The labels is the tokenised target text
        labels = target_tokenized.input_ids
        # The input to the decoder is the labels shifted right, starting with a pad token
        decoder_input_ids = np.full(labels.shape[:-1] + (1,), tokenizer.pad_token_id, labels.dtype)
        decoder_input_ids = np.concatenate([decoder_input_ids, labels[:, :-1]], -1)
        # Replace the padding token id's of the labels with -100, which is the ignore_index of the loss
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized.update(
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=target_tokenized.attention_mask, labels=labels
        )
        return tokenized

    return func


def concat_and_transpose(items):
    return {k: np.stack(item[k] for item in items) for k in items[0].keys()}


def prepare_train_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    dataset = load_dataset("glue", "mnli", split="train")
    dataset = dataset.map(
        form_training_prompts,
        remove_columns=["hypothesis", "premise", "label", "idx"],
        load_from_cache_file=False,
        desc="Generating text prompt",
    )
    dataset = dataset.map(
        tokenizes_text(tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenizing text",
    )
    return dataset


def prepare_validation_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    dataset = load_dataset("glue", "mnli", split="validation_mismatched")
    dataset = dataset.map(
        form_validation_prompts,
        remove_columns=["hypothesis", "premise", "idx"],
        load_from_cache_file=False,
        desc="Generating text prompt",
    )
    dataset = dataset.map(
        prepare_validation_features(tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenizing text",
    )
    return dataset, tokenizer
