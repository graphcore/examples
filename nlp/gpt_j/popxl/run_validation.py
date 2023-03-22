#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging
from functools import partial, reduce
from glob import glob
import re
import os
from typing import Optional, Callable

import torch
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

import wandb

from popxl.utils import to_numpy

from inference import inference
from modelling.embedding import GPTJEmbeddingsTP
from popxl_addons import TaskSession, timer
from utils.setup import gptj_config_setup
from utils.utils import tensor_parallel_input, repeat
from utils.inference import batch_inference
from data.mnli_data import form_validation_prompts, prepare_validation_features, postprocess_mnli_predictions
from config import GPTJConfig
from modelling.hf_mapping import hf_mapping_lm_tp


def unwrap(dl):
    for example in tqdm(dl):
        yield torch.tensor(example["input_ids"], dtype=torch.long)


def run_validation(
    config: GPTJConfig, session: TaskSession, dataset, tokenizer, trained_session: Optional[TaskSession] = None
):
    """
    The session must be opened before calling run_validation
    Usage:
        with session:
            run_validation(config, session, dataset, tokenizer, labels, checkpoint_path)
    """
    if config.checkpoint.load:
        session.load_checkpoint(config.checkpoint.load)
    elif trained_session:
        session.load_from_session(trained_session)

    # else, it assumes weights have been loaded

    labels = dataset["label"]

    tp = config.execution.tensor_parallel
    rf = config.execution.tensor_parallel * config.execution.data_parallel

    def next_token(inputs, lengths):
        data_map = {}
        words = to_numpy(inputs, session.inputs.words.dtype).reshape(-1, *session.inputs.words.shape)
        data_map[session.inputs.words] = tensor_parallel_input(
            words, tp, rf, partial(GPTJEmbeddingsTP.offset_input, config=config)
        ).squeeze()
        data_map[session.inputs.last_token_indices] = repeat(lengths - 1, tp, axis=0)
        # identical for all tp, take first
        next_token_id = session.run(data_map)[session.outputs.next_token][0]
        return torch.LongTensor(next_token_id)

    with timer("Running validation"):
        answers = batch_inference(
            unwrap(dataset),
            next_token,
            config.model.sequence_length,
            eos_token_id=tokenizer.eos_token_id,  # index 50256
            pad_token_id=tokenizer.pad_token_id,  # index 50257
            output_length=config.inference.output_length,
            micro_batch_size=config.execution.micro_batch_size,
        )

    logging.info("Computing validation metric")
    answers = [tokenizer.decode(a, skip_special_tokens=True) for a in answers]
    return answers


def main():
    # --- Config ---
    config, args, _ = gptj_config_setup(
        "config/inference.yml", "release", "gpt-j-mnli", hf_model_setup=False, wandb_setup=False
    )
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.add_special_tokens({"pad_token": "<|extratoken_1|>"})  # index 50257

    # --- Dataset ---
    dataset = load_dataset("glue", "mnli", split="validation_mismatched")
    dataset = dataset.map(
        form_validation_prompts, remove_columns=["hypothesis", "premise", "idx"], load_from_cache_file=False
    )
    dataset = dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # --- Metric ---

    metric = load_metric("glue", "mnli_mismatched")

    # --- Model ---
    max_len = reduce(lambda l, e: max(l, len(e["input_ids"])), dataset, 0)
    config.model.sequence_length = max_len + config.inference.output_length
    logging.info(f"Reducing sequence length to {max_len}")

    session = inference(config)

    pretrained = GPTJForCausalLM.from_pretrained("Graphcore/gptj-mnli")

    with session:
        with timer("Loading HF Graphcore/gptj-mnli model to IPU"):
            session.write_variables_data(hf_mapping_lm_tp(config, session, pretrained))
        answers = run_validation(config, session, dataset, tokenizer)
        formatted_answers = postprocess_mnli_predictions(answers)

    metrics = metric.compute(predictions=formatted_answers, references=dataset["label"])
    logging.info(metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
