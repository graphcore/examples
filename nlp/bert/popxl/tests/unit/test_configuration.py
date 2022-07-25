# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

from config import BertConfig
from utils.simple_parsing_tools import parse_args_with_config_file


def test_grad_accum(test_config_file):
    config = BertConfig.load(test_config_file)
    config.training.global_batch_size = 64
    config.execution.micro_batch_size = 16
    config.execution.data_parallel = 2
    assert config.gradient_accumulation == 64 // (2 * 16)


def test_error_grad_accum(test_config_file):
    config = BertConfig.load(test_config_file)
    config.execution.data_parallel = 3
    with pytest.raises(RuntimeError, match="Unable to set gradient accumulation to match the global batch size"):
        _ = config.gradient_accumulation


def test_defaults_from_config(test_config_file):
    old_defaults = BertConfig()
    new_defaults = parse_args_with_config_file(
        BertConfig, ["--config", test_config_file])
    assert new_defaults != old_defaults


def test_overriden_defaults_from_config(test_config_file):
    default_config = parse_args_with_config_file(
        BertConfig, ["--config", test_config_file])
    assert default_config.execution.data_parallel != 2
    cli_config = parse_args_with_config_file(
        BertConfig, ["--config", test_config_file, "--data_parallel", "2"])
    assert cli_config.execution.data_parallel == 2


def test_bool_arg(test_config_file):
    default_config = BertConfig()
    assert default_config.model.eval is False
    file_config = parse_args_with_config_file(BertConfig, ["--eval"])
    assert file_config.model.eval is True
    empty_config = default_config = parse_args_with_config_file(
        BertConfig, ["--config", test_config_file, "--eval"])
    assert empty_config.model.eval is True
