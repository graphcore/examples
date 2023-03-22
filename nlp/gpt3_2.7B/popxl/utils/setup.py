# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import atexit
import logging
import os
import random
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Callable, Tuple, Union, List

import numpy as np
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel
import popart
import torch
import wandb
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers import GPT2Model

from config import GPTConfig
from popxl_addons import GIT_COMMIT as ADDONS_GIT_COMMIT
from utils.simple_parsing_tools import parse_args_with_presets


def gpt_config_setup(
    config_file: Union[str, Path],
    presets_key: str,
    default: str,
    custom_args: Optional[Callable[[ArgumentParser], None]] = None,
    CLI_args: Optional[str] = None,
) -> Tuple[GPTConfig, argparse.Namespace]:
    def argparse_args(parser: ArgumentParser):
        log_level = os.environ.get("APP_LOG_LEVEL", "INFO")
        parser.add_argument(
            "--log_level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            type=str,
            default=log_level,
            help=("Logging level for the app. " "Can also be set using the environment variable `APP_LOG_LEVEL`"),
        )
        if custom_args:
            custom_args(parser)

    config, args = parse_args_with_presets(GPTConfig, config_file, presets_key, default, argparse_args, CLI_args)
    config: GPTConfig  # type: ignore

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info(f"Staring. Process id: {os.getpid()}")
    logging.info(f"Config: {config}")

    config.validate()

    np.random.seed(config.model.seed)
    torch.manual_seed(config.model.seed)
    random.seed(config.model.seed)

    return config, args


def xl_hf_config_check(config: GPTConfig, hf_config: HFConfig):
    """Compare a GPTConfig with a Hugging Face config and ensure they match. Required if loading a pre-trained model"""
    params = [
        ("hidden_size", config.model.hidden_size, hf_config.n_embd),
        ("heads", config.model.attention.heads, hf_config.n_head),
        ("layers", config.model.layers, hf_config.n_layer),
        ("vocab_size", config.model.embedding.vocab_size, hf_config.vocab_size),
        ("max_positional_length", config.model.embedding.max_positional_length, hf_config.n_positions),
    ]
    if not all(xl == hf for _, xl, hf in params):
        not_eq_str = ", ".join(f"\n`{name}` not equal, config: {xl}, hf: {hf}" for name, xl, hf in params if xl != hf)
        raise ValueError(f"Config does not match the Hugging Face (hf) pre-trained model. Not matching: {not_eq_str}")


def gpt_fine_tuning_setup(
    config_file: Union[str, Path],
    presets_key: str,
    default_config: str,
):
    """Setup config, load pretrained model and check it matches config"""
    config_to_hf = {
        "tiny": "gpt2",  # 12
        "base": "gpt2-medium",  # 24
        "large": "gpt2-large",  # 36
        "xl": "gpt2-xl",  # 48
    }

    def more_args(parser: ArgumentParser):
        parser.add_argument(
            "--hf_model",
            type=str,
            help="HuggingFace transformers pre-trained model to load. "
            "Use 'None' to deliberately skip loading a model for debugging. "
            "If no value is provided it will automatically try and match to the config.",
        )

        parser.add_argument("--wandb", default="True", choices=["False", "True"], help="Initialise Weights and Biases")

    config, args = gpt_config_setup(
        config_file,
        presets_key,
        default_config,
        custom_args=more_args,
    )

    if args.hf_model == "None":
        hf_model = None
    elif args.hf_model is not None:
        hf_model = args.hf_model
    elif args.config in config_to_hf:
        hf_model = config_to_hf[args.config]
    else:
        raise ValueError(
            "Could not match config with `hf_model` automatically. Please provide a hugging face model name or `None`"
        )

    if hf_model:
        pretrained = GPT2Model.from_pretrained(hf_model)
        xl_hf_config_check(config, pretrained.config)
    else:
        pretrained = None

    return config, args, pretrained


def gpt_pretraining_setup(
    config_file: Union[str, Path],
    presets_key: str,
    default_config: str,
):
    """Setup config, load pretrained model and check it matches config"""

    def more_args(parser: ArgumentParser):
        parser.add_argument("--wandb", default="True", choices=["False", "True"], help="Initialise Weights and Biases")

    config, args = gpt_config_setup(
        config_file,
        presets_key,
        default_config,
        custom_args=more_args,
    )

    return config, args


def wandb_init(config: GPTConfig, tags: Optional[List[str]] = None, disable: bool = False):
    """Setup weights and biases"""

    # Save config with addons and popxl version
    config_dict = config.to_dict()
    config_dict["gradient_accumulation"] = config.gradient_accumulation
    config_dict["ipus"] = config.ipus
    config_dict["addons_version"] = ADDONS_GIT_COMMIT
    config_dict["popxl_version"] = popart.versionString()

    mode = "disabled" if disable else "online"

    wandb.init(project="popxl-gpt", tags=tags, config=config_dict, mode=mode)

    # Upload config yml
    # Wandb uploads file asynchronously so can't use a normal context manager
    tmp_dir_cm = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_cm.__enter__()
    atexit.register(lambda: tmp_dir_cm.__exit__(None, None, None))  # Delete directory on exit
    tmp_path = os.path.join(tmp_dir, "config.yml")
    with open(tmp_path, "w") as f:
        config.dump_yaml(f)
    wandb.save(tmp_path, base_path=tmp_dir, policy="now")
