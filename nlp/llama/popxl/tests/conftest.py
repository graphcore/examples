# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
from config import LlamaConfig

import pytest

from utils.simple_parsing_tools import parse_args_with_config_file


def _test_config_file():
    return os.path.join(os.path.dirname(__file__), "test_config.yml")


@pytest.fixture
def test_config_file():
    return _test_config_file()


@pytest.fixture
def test_config():
    return parse_args_with_config_file(LlamaConfig, ["--config", _test_config_file()])


# Below functions enable long tests to be skipped, unless a --long-test
# cli option is specified.
def pytest_addoption(parser):
    parser.addoption("--long-tests", action="store_true", default=False, help="Run long tests")
