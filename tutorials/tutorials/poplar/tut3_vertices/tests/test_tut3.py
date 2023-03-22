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
def test_run_complete():
    """Check that the complete version of the tutorial code runs"""
    testing_util.run_command(
        "../complete/tut3_complete",
        working_path.parent.joinpath("complete"),
        ["Program complete", r"v2: \[7.0000000 6.0000000 4.5000000 2.5000000\]"],
    )
