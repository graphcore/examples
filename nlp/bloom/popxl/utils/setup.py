# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import argparse
import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import popart
import torch
from popxl_addons import GIT_COMMIT as ADDONS_GIT_COMMIT
from popxl_addons import timer
from transformers import AutoTokenizer
from transformers import BloomConfig as HFBloomConfig
from transformers import BloomForCausalLM as HFBloomForCausalLM
from transformers import BloomModel as HFBloomModel

from config import BloomConfig
from utils.simple_parsing_tools import parse_args_with_presets


def bloom_config_setup(
    config_file: Union[str, Path],
    presets_key: str,
    default: str,
    hf_model_setup: bool = False,
    hf_tokenizer_setup: bool = None,
    CLI_args: Optional[str] = None,
) -> Tuple[BloomConfig, argparse.Namespace, Optional[HFBloomModel]]:
    """Parse command line args, setup random seed, W&B, logging and
    load a pre-trained model.

    Args:
        config_file: Path to config file (yaml)
        presets_key: Which key in the config to use
        default: Default model config
        hf_model_setup: Should it add arguments to load an HF pretrained model and load the model if the user
            specifies
        hf_tokenizer_setup: Should HF tokenizer also be loaded. Defaults to same as hf_model_setup, if not set.
        CLI_args: Additional command line arguments.

    Returns:
        BloomConfig, argparse namespace and optional pretrained model
    """
    if hf_tokenizer_setup is None:
        hf_tokenizer_setup = hf_model_setup

    config_to_hf = {
        "bloom_176B_pod16": "bigscience/bloom",
        "bloom_560M_pod16": "bigscience/bloom-560m",
    }

    config_to_cache = {
        "bloom_176B_pod16": "./bloom-ckpt",
        "bloom_560M_pod16": "./bloom-560m-ckpt",
    }

    def custom_args(parser: ArgumentParser):
        log_level = os.environ.get("APP_LOG_LEVEL", "INFO")
        parser.add_argument(
            "--log_level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            type=str,
            default=log_level,
            help=("Logging level for the app. " "Can also be set using the environment variable `APP_LOG_LEVEL`"),
        )

        if hf_model_setup:
            parser.add_argument(
                "--hf_model",
                type=str,
                help="HuggingFace transformers pre-trained model to load. "
                "If no value is provided it will automatically try and match to the config.",
            )

    config, args = parse_args_with_presets(BloomConfig, config_file, presets_key, default, custom_args, CLI_args)
    config: BloomConfig  # type: ignore
    config.validate()

    np.random.seed(config.model.seed)
    torch.manual_seed(config.model.seed)
    random.seed(config.model.seed)

    logging_setup(args, config)

    pretrained = None
    if hf_model_setup and config.execution.memmap_dir is None:
        if args.hf_model is None and args.config in config_to_hf:
            args.hf_model = config_to_hf[args.config]
        elif args.hf_model is None:
            raise ValueError(
                "Could not match config with `hf_model` automatically. "
                "Please provide a hugging face model name, `None` or `Test`."
            )

        with timer("Loading HF model to host"):
            pretrained = HFBloomForCausalLM.from_pretrained(
                args.hf_model,
                cache_dir=config_to_cache[args.config],
                use_auth_token=True,
                torch_dtype=torch.float16,
                offload_state_dict=True,
            )
        logging.info("Finished Loading HF model to host")
        xl_hf_config_check(config, pretrained.config)

    tokenizer = None
    if hf_tokenizer_setup:
        args.hf_model = config_to_hf[args.config]

        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            cache_dir=config_to_cache[args.config],
            use_auth_token=True,
        )

    return config, args, pretrained, tokenizer


def logging_setup(args, config):
    """Setup logging"""
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"Staring. Process id: {os.getpid()}")
    logging.info(f"Config: {config}")


def xl_hf_config_check(config: BloomConfig, hf_config: HFBloomConfig):
    """Compare a BloomConfig with a Hugging Face config and ensure they match. Required if loading a pre-trained model"""
    params = [
        ("hidden_size", config.model.hidden_size, hf_config.hidden_size),
        ("heads", config.model.attention.heads, hf_config.n_head),
        ("layers", config.model.layers, hf_config.n_layer),
        ("vocab_size", config.model.embedding.vocab_size, hf_config.vocab_size),
    ]
    if not all(xl == hf for _, xl, hf in params):
        not_eq_str = ", ".join(f"\n`{name}` not equal, config: {xl}, hf: {hf}" for name, xl, hf in params if xl != hf)
        raise ValueError(f"Config does not match the Hugging Face (hf) pre-trained model. Not matching: {not_eq_str}")
