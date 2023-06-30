# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os
import numpy as np
from typing import Dict, Any
import torch
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
from config import GPTConfig
from popxl_addons.array_munging import pad_axis

## Generative prompts
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


## Classification head
def form_training_prompts(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]
    example["text"] = f"{hypothesis}<|sep|>{premise}<|cls|>".strip()
    example["class_label"] = ["entailment", "neutral", "contradiction"][example["label"]]
    return example


def tokenize(dataset: Dataset, tokenizer: GPT2Tokenizer, max_length: int):
    tokenized = tokenizer.encode(
        dataset["text"],
        return_attention_mask=False,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=max_length,
    ).flatten()
    length = len(tokenized)
    tokenized = pad_axis(tokenized, max_length, axis=0, value=tokenizer.pad_token_id)
    tokenized = {"input_ids": tokenized, "unpadded_length": length}
    return tokenized


def concat_fnc(items):
    return {
        "input_ids": np.stack([item["input_ids"] for item in items]),
        "label": np.stack([item["label"] for item in items]),
        "unpadded_length": np.stack([item["unpadded_length"] for item in items]),
        "text": [item["text"] for item in items],
        "class_label": [item["class_label"] for item in items],
    }


def prepare_dataset(config: GPTConfig, split: str):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})  # index 50257
    tokenizer.add_special_tokens({"sep_token": "<|sep|>"})  # index 50258
    tokenizer.add_special_tokens({"cls_token": "<|cls|>"})  # index 50259

    dataset = load_dataset("glue", "mnli", split=split)
    dataset_extra = dataset.map(
        form_training_prompts,
        remove_columns=["hypothesis", "premise", "idx"],
        load_from_cache_file=True,
        desc="Generating text prompt",
    )
    np.object = object  # TODO due to bug in pyarrow
    dataset_token = dataset_extra.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config.model.sequence_length,
        },  # Specify here so included in cache
        num_proc=1,
        load_from_cache_file=True,
        desc="Tokenizing text",
    )
    return dataset_token
