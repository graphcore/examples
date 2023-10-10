#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from typing import Optional, Tuple

from transformers import AutoTokenizer
from utils.setup import llama_config_setup, supported_ckpts
from config import LlamaConfig
from api import LlamaPipeline


def run_inference_popxl(config: LlamaConfig, tokenizer, hf_model, sequence_length: Optional[int] = None):
    if sequence_length is not None:
        config.model.sequence_length = sequence_length

    pipe = LlamaPipeline(config, hf_llama_checkpoint=hf_model, tokenizer=tokenizer)

    def get_input() -> Tuple[str, float, int, int]:
        while True:
            try:
                logging.info("-- Enter prompt --")
                prompt = input("> ")
                logging.info("-- Enter Sampling Temperature (0 for greedy) --")
                temperature = float(input("> "))
                logging.info("-- Enter top-k parameter (0 for max) --")
                k = int(input("> "))
                logging.info("-- Enter number of tokens to generate --")
                num_tokens = int(input("> "))
                break
            except ValueError:
                logging.info("Invalid input!")

        return prompt, temperature, k, num_tokens

    while True:
        prompt, temperature, k, output_length = get_input()
        pipe(prompt, k=k, temperature=temperature, output_length=output_length)[0]


def main():
    # --- Setup ---
    default_config = "llama2_70b_pod16"
    config, _, hf_model = llama_config_setup("config/inference.yml", "release", default_config, hf_model_setup=True)

    tokenizer = AutoTokenizer.from_pretrained(supported_ckpts[default_config])

    run_inference_popxl(config, tokenizer, hf_model=hf_model, sequence_length=config.model.sequence_length)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
