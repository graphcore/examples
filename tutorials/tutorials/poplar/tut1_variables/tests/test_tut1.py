# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path

import pytest
from filelock import FileLock

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent


@pytest.fixture(autouse=True)
def with_compiled_examples():
    """Compile the complete version of the tutorial code"""
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("make all", working_path)


@pytest.mark.category1
def test_run_ipu_model():
    """Check that the complete version of the tutorial code
    for the IPU Model runs"""
    testing_util.run_command(
        "./tut1_ipu_model_complete",
        working_path,
        [
            "Program complete",
            "h3 data:",
            "0 1 1.5 2",
            r"v4-1: \[10 11 12 13 14 15 16 17 18 19\]",
        ],
    )


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_run_ipu_hardware():
    """Check that the complete version of the tutorial code
    for the IPU hardware runs"""
    testing_util.run_command(
        "./tut1_ipu_hardware_complete",
        working_path,
        [
            "Attached to IPU",
            "Program complete",
            "h3 data:",
            "0 1 1.5 2",
            r"v4-1: \[10 11 12 13 14 15 16 17 18 19\]",
        ],
    )
