# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

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

from popxl_addons import GIT_COMMIT as ADDONS_GIT_COMMIT
from popxl_addons.utils import timer

from configs import MagmaConfig
from utils.simple_parsing_tools import parse_args_with_presets
import popdist
import sys

__all__ = ["set_random_seeds", "magma_config_setup"]


def set_random_seeds(seed: int) -> None:
    """
    Initialise seeds on host (numpy, torch, random)
    to guarantee deterministic results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def magma_config_setup(
    config_file: Union[str, Path],
    presets_key: str,
    default: str,
    CLI_args: Optional[str] = None,
) -> Tuple[MagmaConfig, argparse.Namespace]:
    """Parse command line args and setup random seed, W&B, logging
    Args:
        config_file: Path to config file (yaml)
        presets_key: Which key in the config to use
        default: Default model config
        CLI_args: Extra command line arguments to customise configuration

    Returns:
        MagmaConfig, argparse namespace and optional pretrained model
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
        # needed for jupyter notebooks
        parser.add_argument("-f", type=str, default="", help=f"jupyter")

    config, args = parse_args_with_presets(MagmaConfig, config_file, presets_key, default, custom_args, CLI_args)
    config: MagmaConfig  # type: ignore
    config.validate()

    set_random_seeds(config.seed)

    logging_setup(args, config)

    return config, args


def logging_setup(args, config):
    """Setup logging"""
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.info(f"Starting. Process id: {os.getpid()}")
    logging.info(f"Config: {config}")
