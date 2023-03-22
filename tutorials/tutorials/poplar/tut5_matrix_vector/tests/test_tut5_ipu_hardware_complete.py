# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent.joinpath("complete")


@pytest.fixture(autouse=True)
def with_compiled_example():
    """Compile the IPU version of the tutorial code"""
    testing_util.run_command("make tut5_ipu", working_path)


@pytest.mark.category1
def test_run_complete():
    """Check that the complete version of the tutorial code runs"""
    testing_util.run_command(
        "./tut5_ipu 40 50",
        working_path,
        [
            "Multiplying matrix of size 40x50 by vector of size 50",
            "Multiplication result OK",
        ],
    )
