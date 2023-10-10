# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
from typing import Dict, Any
from transformers import default_data_collator
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from .hf_data_utils import group_texts


def form_training_prompts(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]
    class_label = ["entailment", "neutral", "contradiction"][example["label"]]

    example["text"] = f"mnli hypothesis: {hypothesis} premise: {premise} target: {class_label}<|endoftext|>"
    return example


def form_validation_prompts(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]

    example["text"] = f"mnli hypothesis: {hypothesis} premise: {premise} target:"
    return example


def prepare_validation_features(dataset, tokenizer):
    tokenized_examples = []
    for example in dataset["text"]:
        tokenized_example = tokenizer.encode(example, return_tensors="pt").squeeze()
        tokenized_examples.append(tokenized_example)
    return {"input_ids": tokenized_examples, "label": dataset["label"]}


def extract_class_label(s: str) -> str:
    """Extract a class label from the generated text.
    This is done by matching the label as there is often no space between the label and subsequent output."""
    s = s.strip()
    # no need if decoded using skip special tokens
    s = s.replace("<|endoftext|>", "")
    class_labels = ["entailment", "neutral", "contradiction"]
    if s in class_labels:
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


class PadCollate:
    """
    Collate into a batch and pad the batch up to a fixed size.
    """

    def __init__(self, batch_size, padding_val_dict=None):
        self.batch_size = batch_size
        self.padding_val_dict = padding_val_dict

    def pad_tensor(self, x, val):
        pad_size = list(x.shape)
        pad_size[0] = self.batch_size - x.size(0)
        return torch.cat([x, val * torch.ones(*pad_size, dtype=x.dtype)], dim=0)

    def __call__(self, items):
        batch = default_data_collator(items)
        if len(items) < self.batch_size:
            for k in batch.keys():
                batch[k] = self.pad_tensor(batch[k], self.padding_val_dict[k])
        return batch


def tokenizes_text(tokenizer):
    def func(dataset):
        tokenized = tokenizer(dataset["text"], return_attention_mask=False, return_tensors="np")
        return tokenized

    return func


def concat_and_transpose(items):
    return {k: np.stack(item[k] for item in items) for k in ["input_ids", "labels"]}


def prepare_train_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
    tokenizer.add_special_tokens({"pad_token": "<|extratoken_1|>"})  # index 50257

    dataset = load_dataset("glue", "mnli", split="train")
    dataset = dataset.map(
        form_training_prompts,
        remove_columns=["hypothesis", "premise", "idx"],
        load_from_cache_file=True,
        desc="Generating text prompt",
    )
    dataset = dataset.map(
        tokenizes_text(tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing text",
    )
    dataset = dataset.map(
        group_texts(config),
        batched=True,
        batch_size=1000,
        num_proc=1,
        load_from_cache_file=True,
        desc="Packing sequences",
    )
    return dataset
