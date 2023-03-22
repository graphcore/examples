#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging
from functools import partial, reduce

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GPTJForCausalLM

import popxl
from popxl.utils import to_numpy

from inference import inference
from modelling.embedding import GPTJEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp
from popxl_addons import timer
from utils.setup import gptj_config_setup
from utils.utils import tensor_parallel_input, repeat
from utils.inference import batch_inference
from data.mnli_data import form_validation_prompts, prepare_validation_features
from config import GPTJConfig


def unwrap(dl):
    for example in tqdm(dl):
        yield torch.tensor(example["input_ids"], dtype=torch.long)


def run_inference_hf(dataset, tokenizer, hf_model, sequence_length, output_length, micro_batch_size):
    logging.info("Running inference HF")

    def next_token(inputs, lengths):
        out = hf_model(input_ids=inputs).logits  # Tensor[mbs, seq, vocab]
        # Batched index_select:
        #   Flatten [mbs, seq] dimension and offset indices
        mbs = out.shape[0]
        seq = out.shape[1]
        offsets = (lengths - 1) + (torch.arange(0, mbs) * seq)
        next_token_logits = torch.index_select(out.reshape(-1, *out.shape[2:]), 0, offsets)  # Tensor[mbs, vocab]
        return torch.argmax(next_token_logits, dim=-1).reshape(-1)

    answers = batch_inference(
        unwrap(dataset),
        next_token,
        sequence_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_length=output_length,
        micro_batch_size=micro_batch_size,
    )

    logging.info("HF output")
    for a in answers:
        text = tokenizer.decode(a)
        logging.info(f"detokenized: {text}")


def run_inference_popxl(config: GPTJConfig, dataset, tokenizer, hf_model, sequence_length, output_length):
    config.model.sequence_length = sequence_length
    tp = config.execution.tensor_parallel
    rf = config.execution.tensor_parallel * config.execution.data_parallel

    session = inference(config)

    if config.model.dtype == popxl.float16:
        hf_model.half()
    with timer("Loading HF pretrained model to IPU"):
        weights = hf_mapping_lm_tp(config, session, hf_model)
        session.write_variables_data(weights)

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

    logging.info("Attach to IPUs")
    with session:
        answers = batch_inference(
            unwrap(dataset),
            next_token,
            config.model.sequence_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_length=output_length,
            micro_batch_size=config.execution.micro_batch_size,
        )

    logging.info("popxl output")
    for a in answers:
        text = tokenizer.decode(a)
        logging.info(f"detokenized: {text}")


def main():
    # --- Setup ---
    config, *_ = gptj_config_setup("config/inference.yml", "release", "gpt-j-mnli")
    # --- Tokenizer ---
    hf_model = GPTJForCausalLM.from_pretrained("Graphcore/gptj-mnli")
    tokenizer = AutoTokenizer.from_pretrained("Graphcore/gptj-mnli")
    tokenizer.add_special_tokens({"pad_token": "<|extratoken_1|>"})  # index 50257
    # --- Dataset ---
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

    max_len = reduce(lambda l, e: max(l, len(e["input_ids"])), dataset, 0)
    output_length = 5

    # --- HF example ---
    logging.info("Initialising HF model")
    hf_model.eval()
    run_inference_hf(
        dataset, tokenizer, hf_model, max_len + output_length, output_length, config.execution.micro_batch_size
    )

    # --- POPXL example ---
    run_inference_popxl(config, dataset, tokenizer, hf_model, max_len + output_length, output_length)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
