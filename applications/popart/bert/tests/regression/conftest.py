# Copyright 2019 Graphcore Ltd.
import os
from pathlib import Path
import pytest


def pytest_addoption(parser):
    bert_dir = Path(__file__).parent.parent.parent.resolve()

    default_output_path = str(bert_dir / "test_results")

    parser.addoption(
        "--output-path",
        action="store",
        help="Path to a directory into which test results should be written " +
        "(will be created if it doesn't exist).",
        default=default_output_path)


@pytest.fixture
def output_path(request):
    return request.config.getoption("--output-path")
