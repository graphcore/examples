# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import argparse
import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import numpy as np

from transformers import LlamaForCausalLM, LlamaConfig as HFConfig

import popxl
from popxl_addons.utils import timer

from config import LlamaConfig
from utils.simple_parsing_tools import parse_args_with_presets
import sys

supported_ckpts = {
    "llama2_7b_pod4": "meta-llama/Llama-2-7b-chat-hf",
    "llama2_13b_pod4": "meta-llama/Llama-2-13b-chat-hf",
    "llama2_70b_pod16": "meta-llama/Llama-2-70b-chat-hf",
}

supported_ckpts["llama2_7b_pod2"] = supported_ckpts["llama2_7b_pod4"]
supported_ckpts["llama2_7b_pod16"] = supported_ckpts["llama2_7b_pod4"]
supported_ckpts["llama2_13b_pod16"] = supported_ckpts["llama2_13b_pod4"]
supported_ckpts["llama2_70b_pod64"] = supported_ckpts["llama2_70b_pod16"]


def llama_config_setup(
    config_file: Union[str, Path],
    presets_key: str,
    default: str,
    hf_model_setup=False,
    CLI_args: Optional[str] = None,
) -> Tuple[LlamaConfig, argparse.Namespace, Optional[LlamaForCausalLM]]:
    """Parse command line args and setup random seed, W&B, logging and
    load a pre-trained model.

    Args:
        config_file: Path to config file (yaml)
        presets_key: Which key in the config to use
        default: Default model config
        hf_model_setup: Should it add arguments to load an HF pretrained model and load the model if the user
            specifies
        CLI_args:

    Returns:
        LlamaConfig, argparse namespace and optional pretrained model
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

        # needed for jupyter notebooks
        parser.add_argument("-f", type=str, default="", help=f"jupyter")

    config, args = parse_args_with_presets(LlamaConfig, config_file, presets_key, default, custom_args, CLI_args)
    config: LlamaConfig  # type: ignore
    config.validate()

    if config.execution.use_cache:
        assert config.execution.micro_batch_size == 1, "`micro_batch_size` > 1 not supported when using caching."

    np.random.seed(config.model.seed)
    torch.manual_seed(config.model.seed)
    random.seed(config.model.seed)

    logging_setup(args, config)

    if hf_model_setup:
        if args.hf_model == "None":
            pretrained = None
        elif args.hf_model == "Test":
            hf_config = HFConfig(
                vocab_size=config.model.embedding.vocab_size,
                max_position_embeddings=config.model.sequence_length,
                hidden_size=config.model.hidden_size,
                intermediate_size=config.model.intermediate_size,
                num_hidden_layers=config.model.layers,
                num_attention_heads=config.model.attention.heads,
            )
            pretrained = LlamaForCausalLM(hf_config)
        else:
            try:
                hf_model = supported_ckpts[default]
            except Exception as e:
                raise e

            with timer("Loading HF model to host"):
                if default in ["llama2_70b_pod16"]:
                    if config.model.dtype != popxl.float16:
                        logging.warn(
                            "Model dtype specified does not match 70b model weights which must be loaded in FP16."
                        )

                    pretrained = LlamaForCausalLM.from_pretrained(
                        hf_model, low_cpu_mem_usage=True, torch_dtype=torch.float16
                    )
                else:
                    pretrained = LlamaForCausalLM.from_pretrained(hf_model)

            xl_hf_config_check(config, pretrained.config)
    else:
        pretrained = None

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


def xl_hf_config_check(config: LlamaConfig, hf_config: HFConfig):
    """Compare a LlamaConfig with a Hugging Face config and ensure they match. Required if loading a pre-trained model"""
    params = [
        ("hidden_size", config.model.hidden_size, hf_config.hidden_size),
        ("heads", config.model.attention.heads, hf_config.num_attention_heads),
        ("layers", config.model.layers, hf_config.num_hidden_layers),
        ("vocab_size", config.model.embedding.vocab_size, hf_config.vocab_size),
    ]
    if not all(xl == hf for _, xl, hf in params):
        not_eq_str = ", ".join(f"\n`{name}` not equal, config: {xl}, hf: {hf}" for name, xl, hf in params if xl != hf)
        raise ValueError(f"Config does not match the Hugging Face (hf) pre-trained model. Not matching: {not_eq_str}")
