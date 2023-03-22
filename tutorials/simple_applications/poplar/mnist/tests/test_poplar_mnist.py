# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from pathlib import Path

import pytest
from filelock import FileLock

# NOTE: The imports below are dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def setUp():
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("make all", working_path, [])


@pytest.fixture(autouse=True)
def download_mnist():
    files = [
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ]
    if not all([(working_path / f"/data/{f}").exists() for f in files]):
        testing_util.run_command("./get_data.sh", working_path)


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_ipu_model():
    """Check that the tutorial code runs on the IPU Model"""
    testing_util.run_command(
        "./regression-demo 1 1.0",
        working_path,
        ["Using the IPU Model", "Epoch", "100%"],
    )


@pytest.mark.category2
def test_ipu_hardware():
    """Check that the tutorial code runs on the IPU hardware"""
    testing_util.run_command(
        "./regression-demo -IPU 1 1.0",
        working_path,
        ["Using the IPU", "Epoch", "100%"],
    )
