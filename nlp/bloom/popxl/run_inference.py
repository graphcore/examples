# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
import os
import time
from typing import Optional

import numpy as np
from popxl_addons.utils import timer
from transformers import AutoModel, AutoTokenizer
from scipy.special import softmax

from config import CONFIG_DIR, BloomConfig
from inference import inference
from modelling.bloom_lm import BloomLMHeadModelTP2D
from utils.setup import bloom_config_setup
from popxl_addons import TaskSession

from popxl_addons.array_munging import repeat

# Creates the session and writes weights to IPU (if `hf_model` is not `None`)
def setup_session(config: BloomConfig, hf_model: Optional[AutoModel] = None) -> TaskSession:
    session = inference(config)

    msg = "Must pass precisely one of `hf_model` or `config.execution.memmap_dir`"
    assert bool(hf_model) != bool(config.execution.memmap_dir), msg
    if hf_model:
        with timer("Loading HF pretrained model to IPU"):
            weights = BloomLMHeadModelTP2D.hf_mapping(config, session.state, hf_model)
            session.write_variables_data(weights)

    return session


# Performs one model forward pass on `inputs`, samples next token, and returns it.
# Typically called in a loop, incrementing `lengths` by 1 each iteration.
def next_token(
    session: TaskSession,
    inputs: np.array,
    lengths: np.array,
    tp1: int,
    tp2: int,
    temperature: float = 1.0,
    k: Optional[int] = 3,
):
    next_token_logits = session.run(
        {
            session.inputs.words: repeat(inputs, tp1 * tp2, axis=0),
            session.inputs.last_token_indices: repeat(np.array(lengths - 1), tp1 * tp2),
        }
    )[session.outputs.next_token_logits][0]

    # partition logits to top k results.
    if k:
        topk_idx = np.argpartition(next_token_logits, -k)[-k:]
        next_token_logits = next_token_logits[topk_idx]

    if temperature > 0:
        next_token_prob = softmax(next_token_logits.astype(np.float32) / temperature)
        next_token_id = np.random.choice(next_token_logits.shape[-1], p=next_token_prob)
    else:  # equivalent to temperature = 0, but argmax is faster.
        next_token_id = next_token_logits.argmax()

    if k:
        next_token_id = topk_idx[next_token_id]

    return next_token_id


# Performs initial tokenization and padding of the input prompt
def tokenize_initial(prompt: str, tokenizer: AutoTokenizer, config: BloomConfig):
    tokenizer.padding_side = "right"
    tokenizer_result = tokenizer(prompt, return_length=True, return_tensors="np")
    tokenized_prompt = tokenizer_result.input_ids[0]
    tokenized_length = tokenizer_result.length

    padded_prompt = np.full((config.model.sequence_length,), tokenizer.pad_token_id)
    padded_prompt[: tokenized_length[0]] = tokenized_prompt

    return padded_prompt, tokenized_prompt, tokenized_length


def run_inference_popxl(
    config: BloomConfig,
    tokenizer: AutoTokenizer,
    hf_model: Optional[AutoModel] = None,
):
    tp1, tp2 = config.execution.tensor_parallel_1, config.execution.tensor_parallel_2

    session = setup_session(config, hf_model)

    logging.info("Attaching to IPUs")
    # Begin interactive loop
    with session:
        while True:
            flag = True
            while flag:
                try:
                    logging.info("-- Enter prompt --")
                    prompt = input("> ")
                    logging.info("-- Enter Sampling Temperature (0 for greedy) --")
                    temperature = float(input("> "))
                    logging.info("-- Enter top-k parameter (0 for max) --")
                    k = int(input("> "))
                    logging.info("-- Enter number of tokens to generate --")
                    num_tokens = int(input("> "))
                    flag = False
                except ValueError:
                    logging.info("Invalid input!")

            padded_prompt, tokenized_prompt, tokenized_length = tokenize_initial(prompt, tokenizer, config)

            num_generated = 0
            result = tokenized_prompt.tolist()

            # Begin inference loop
            logging.info("Beginning inference loop")
            print(tokenizer.decode(result), end="", flush=True)
            start_time = time.time()
            for _ in range(num_tokens):
                t = next_token(session, padded_prompt, tokenized_length[0], tp1, tp2, temperature, k)
                result.append(t)
                print(tokenizer.decode(t), end="", flush=True)
                padded_prompt[tokenized_length[0]] = t
                tokenized_length[0] += 1
                num_generated += 1

                if result[-1] == tokenizer.eos_token_id or tokenized_length[0] >= config.model.sequence_length:
                    break
            print("")
            end_time = time.time()

            logging.info(f"Output in {end_time - start_time:.2f} seconds")
            logging.info(f"Throughput: {num_generated / (end_time - start_time):.2f} t/s")


def main():
    # Configuration
    config, args, pretrained, tokenizer = bloom_config_setup(
        CONFIG_DIR / "inference.yml",
        "release",
        "bloom_560M",
        hf_model_setup=True,
        hf_tokenizer_setup=True,
    )

    if pretrained:
        pretrained = pretrained.eval()

    run_inference_popxl(config, tokenizer, hf_model=pretrained)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
