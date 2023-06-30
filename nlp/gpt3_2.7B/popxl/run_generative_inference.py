#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from functools import reduce
from copy import deepcopy

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer

import popxl
from popxl.utils import to_numpy
from popxl_addons.array_munging import tensor_parallel_input, repeat

from generative_inference import generative_inference
from modelling.hf_mapping import hf_mapping_lm_gen_inference_TP
from popxl_addons import timer
from utils.setup import gpt_config_setup
from utils.inference import batch_inference
from data.mnli.mnli_data import form_validation_prompts, prepare_validation_features
from config import GPTConfig


def unwrap(dl):
    for example in tqdm(dl):
        yield torch.tensor(example, dtype=torch.long)


def run_inference_hf(dataset, tokenizer, hf_model, sequence_length, output_length, micro_batch_size):
    logging.info("Running inference HF")

    def next_token(inputs, lengths):
        outputs = hf_model(input_ids=inputs)
        logits = outputs.logits  # Tensor[mbs, seq, vocab]
        # Batched index_select:
        #   Flatten [mbs, seq] dimension and offset indices
        mbs = logits.shape[0]
        seq = logits.shape[1]
        offsets = (lengths - 1) + (torch.arange(0, mbs) * seq)
        next_token_logits = torch.index_select(logits.reshape(-1, *logits.shape[2:]), 0, offsets)  # Tensor[mbs, vocab]
        return torch.argmax(next_token_logits, dim=-1).reshape(-1)

    answers = batch_inference(
        unwrap(dataset),
        next_token,
        sequence_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=0,
        output_length=output_length,
        micro_batch_size=micro_batch_size,
    )

    logging.info("HF output")
    for p, a in zip(dataset, answers):
        prompt = tokenizer.decode(p)
        answer = tokenizer.decode(a)
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Answer: {answer}")


def run_inference_popxl(config: GPTConfig, dataset, tokenizer, hf_model, sequence_length, output_length):
    config.model.sequence_length = sequence_length
    tp = config.execution.tensor_parallel
    rf = config.execution.tensor_parallel * config.execution.data_parallel

    session = generative_inference(config)
    if config.model.dtype == popxl.float16:
        hf_model.half()
    with timer("Loading HF pretrained model to IPU"):
        weights = hf_mapping_lm_gen_inference_TP(config, session, hf_model)
        session.write_variables_data(weights)

    def next_token(inputs, lengths):
        data_map = {}
        words = to_numpy(inputs, session.inputs.words.dtype).reshape(-1, *session.inputs.words.shape)
        data_map[session.inputs.words] = tensor_parallel_input(words, tp, rf).squeeze()
        data_map[session.inputs.last_token_indices] = repeat(lengths - 1, tp, axis=0)
        outputs = session.run(data_map)
        next_token_id = outputs[session.outputs.next_token][0]  # identical for all tp, take first
        return torch.LongTensor(next_token_id)

    logging.info("Attach to IPUs")
    with session:
        answers = batch_inference(
            unwrap(dataset),
            next_token,
            config.model.sequence_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=0,
            output_length=output_length,
            micro_batch_size=config.execution.micro_batch_size,
        )

    logging.info("popxl output")
    for p, a in zip(dataset, answers):
        prompt = tokenizer.decode(p)
        try:
            answer = tokenizer.decode(a)
        except TypeError as e:
            raise Exception(f"Couldn't de-tokenize: {a}") from e
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Answer: {answer}")


def get_mnli_dataset(config, tokenizer):
    """MNLI dataset"""
    dataset = load_dataset("glue", "mnli", split="validation_mismatched")
    dataset = dataset.select(range(config.execution.micro_batch_size))
    dataset = dataset.map(form_validation_prompts, remove_columns=["hypothesis", "premise", "idx"])
    dataset = dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    dataset = [e["input_ids"] for e in dataset]
    return dataset


def get_dummy_dataset(config, tokenizer):
    """Dummy dataset"""
    text = [
        "Marry had a little ",
        "Edinburg is the capital of ",
        "My name is ",
    ]
    dataset = [tokenizer.encode(t, return_tensors="pt").flatten() for t in text]
    return dataset


def main():
    # --- Setup ---
    config, args, pretrained = gpt_config_setup(
        "config/inference.yml", "release", "gpt2_small", wandb_setup=False, hf_model_setup=True
    )

    # --- Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    assert (
        config.model.embedding.vocab_size >= tokenizer.vocab_size
    ), f"Vocab size of model is too small for tokenizer: {config.model.embedding.vocab_size} < {tokenizer.vocab_size}"
    # --- Dataset ---
    dataset = list(get_dummy_dataset(config, tokenizer))  # Should just be input_ids

    max_len = reduce(lambda l, e: max(l, len(e)), dataset, 0)
    output_length = config.inference.generative_output_len

    # --- HF example ---
    if pretrained:
        logging.info("Initialising HF model")
        pretrained.eval()
        run_inference_hf(
            deepcopy(dataset),
            tokenizer,
            pretrained,
            max_len + output_length,
            output_length,
            config.execution.micro_batch_size,
        )

    # --- POPXL example ---
    run_inference_popxl(config, dataset, tokenizer, pretrained, max_len + output_length, output_length)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
