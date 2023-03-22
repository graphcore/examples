# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
import pytest
from filelock import FileLock

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def with_compiled_examples():
    """Compile the tutorial code"""
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("make all", working_path)


@pytest.mark.category1
def test_run_complete_ipu_hardware():
    """Check that the tutorial code runs on IPU hardware"""
    testing_util.run_command(
        "./tut6 10000 1000 --device ipu",
        working_path,
        [
            "Multiplying matrix of size 10000x1000 by vector of size 1000",
            "Worst cost seen: 53807",
            "Multiplication result OK",
        ],
    )


@pytest.mark.category1
def test_run_complete_mk1():
    """Check that the tutorial code runs on Mk1"""
    testing_util.run_command(
        "./tut6 10000 1000 --device model-ipu1",
        working_path,
        [
            "Multiplying matrix of size 10000x1000 by vector of size 1000",
            "Worst cost seen: 64373",
            "Multiplication result OK",
        ],
    )


@pytest.mark.category1
def test_run_complete_mk2():
    """Check that the tutorial code runs on Mk2"""
    testing_util.run_command(
        "./tut6 10000 1000 --device model-ipu2",
        working_path,
        [
            "Multiplying matrix of size 10000x1000 by vector of size 1000",
            "Worst cost seen: 53807",
            "Multiplication result OK",
        ],
    )
