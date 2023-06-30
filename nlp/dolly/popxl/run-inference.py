#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from typing import Optional, Tuple
import time

from transformers import AutoTokenizer
from scipy.special import softmax

from inference import inference
from modelling.embedding import DollyEmbeddingsTP
from modelling.hf_mapping import hf_mapping_lm_tp
from popxl_addons import timer
from utils.setup import dolly_config_setup
from popxl_addons.array_munging import tensor_parallel_input, repeat
from config import DollyConfig
from api import DollyPipeline

import numpy as np


def run_inference_popxl(config: DollyConfig, tokenizer, hf_model, sequence_length: Optional[int] = None):
    if sequence_length is not None:
        config.model.sequence_length = sequence_length

    pipe = DollyPipeline(config, hf_dolly_checkpoint=hf_model, tokenizer=tokenizer)

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
    config, _, hf_model = dolly_config_setup("config/inference.yml", "release", "dolly_pod16", hf_model_setup=True)
    config.execution.micro_batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

    run_inference_popxl(config, tokenizer, hf_model=hf_model, sequence_length=512)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
