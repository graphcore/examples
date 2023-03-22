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
import popart
import torch
import wandb
from transformers.models.gptj import GPTJConfig as HFConfig
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

from popxl_addons import GIT_COMMIT as ADDONS_GIT_COMMIT
from popxl_addons.utils import timer

from config import GPTJConfig
from utils.simple_parsing_tools import parse_args_with_presets
import popdist
import sys


def gptj_config_setup(
    config_file: Union[str, Path],
    presets_key: str,
    default: str,
    wandb_setup=False,
    hf_model_setup=False,
    CLI_args: Optional[str] = None,
) -> Tuple[GPTJConfig, argparse.Namespace, Optional[GPTJForCausalLM]]:
    """Parse command line args and setup random seed, W&B, logging and
    load a pre-trained model.

    Args:
        config_file: Path to config file (yaml)
        presets_key: Which key in the config to use
        default: Default model config
        wandb_setup: Should it initialise Weights and Biases
        hf_model_setup: Should it add arguments to load an HF pretrained model and load the model if the user
            specifies
        CLI_args:

    Returns:
        GPTJConfig, argparse namespace and optional pretrained model
    """

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
                "Use 'None' to deliberately skip loading a model for debugging. "
                "If no value is provided it will automatically try and match to the config.",
            )

        if wandb_setup:
            parser.add_argument(
                "--wandb", default="True", choices=["False", "True"], help="Initialise Weights and Biases"
            )
        # needed for jupyter notebooks
        parser.add_argument("-f", type=str, default="", help=f"jupyter")

    config, args = parse_args_with_presets(GPTJConfig, config_file, presets_key, default, custom_args, CLI_args)
    config: GPTJConfig  # type: ignore
    config.validate()

    np.random.seed(config.model.seed)
    torch.manual_seed(config.model.seed)
    random.seed(config.model.seed)

    if wandb_setup:
        if popdist.getInstanceIndex() == 0:
            wandb_init(config, tags=["PE", "TP"], disable=args.wandb == "False")

    logging_setup(args, config)

    if hf_model_setup:
        if args.hf_model == "None":
            pretrained = None
        elif args.hf_model == "Test":
            hf_config = HFConfig(
                vocab_size=config.model.embedding.vocab_size,
                n_positions=config.model.sequence_length,
                n_embd=config.model.hidden_size,
                n_layer=config.model.layers,
                n_head=config.model.attention.heads,
                rotary_dim=config.model.attention.rotary_dim,
                resid_pdrop=config.model.dropout_prob,
                embd_pdrop=config.model.dropout_prob,
                attn_pdrop=config.model.dropout_prob,
            )
            pretrained = GPTJForCausalLM(hf_config)
        else:
            hf_model = "EleutherAI/gpt-j-6B"
            with timer("Loading HF model to host"):
                pretrained = GPTJForCausalLM.from_pretrained(hf_model)
            xl_hf_config_check(config, pretrained.config)
    else:
        pretrained = None

    return config, args, pretrained


def gptj_fine_tuning_setup(
    config_file: Union[str, Path], presets_key: str, default_config: str, wandb_setup: bool = False
) -> Tuple[GPTJConfig, argparse.Namespace, Optional[GPTJForCausalLM]]:
    """GPT-J setup for finetuning scripts"""
    config, args, pretrained = gptj_config_setup(
        config_file, presets_key, default_config, wandb_setup=wandb_setup, hf_model_setup=True
    )

    return config, args, pretrained


def logging_setup(args, config):
    """Setup logging"""
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.info(f"Starting. Process id: {os.getpid()}")


def xl_hf_config_check(config: GPTJConfig, hf_config: HFConfig):
    """Compare a GPTJConfig with a Hugging Face config and ensure they match. Required if loading a pre-trained model"""
    params = [
        ("hidden_size", config.model.hidden_size, hf_config.n_embd),
        ("heads", config.model.attention.heads, hf_config.n_head),
        ("layers", config.model.layers, hf_config.n_layer),
        ("vocab_size", config.model.embedding.vocab_size, hf_config.vocab_size),
        ("rotary_dim", config.model.attention.rotary_dim, hf_config.rotary_dim),
    ]
    if not all(xl == hf for _, xl, hf in params):
        not_eq_str = ", ".join(f"\n`{name}` not equal, config: {xl}, hf: {hf}" for name, xl, hf in params if xl != hf)
        raise ValueError(f"Config does not match the Hugging Face (hf) pre-trained model. Not matching: {not_eq_str}")


def wandb_init(config: GPTJConfig, tags: Optional[List[str]] = None, disable: bool = False):
    """Setup weights and biases"""
    # Save config with addons and popxl version
    config_dict = config.to_dict()
    config_dict["gradient_accumulation"] = config.gradient_accumulation
    config_dict["ipus"] = config.ipus
    config_dict["addons_version"] = ADDONS_GIT_COMMIT
    config_dict["popxl_version"] = popart.versionString()

    mode = "disabled" if disable else "online"

    wandb.init(project="popxl-gptj", tags=tags, config=config_dict, mode=mode)

    # Upload config yml
    # Wandb uploads file asynchronously so can't use a normal context manager
    tmp_dir_cm = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_cm.__enter__()
    atexit.register(lambda: tmp_dir_cm.__exit__(None, None, None))  # Delete directory on exit
    tmp_path = os.path.join(tmp_dir, "config.yml")
    with open(tmp_path, "w") as f:
        config.dump_yaml(f)
    wandb.save(tmp_path, base_path=tmp_dir, policy="now")
