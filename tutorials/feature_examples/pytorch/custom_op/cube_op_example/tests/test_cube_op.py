# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
from filelock import FileLock
from pathlib import Path

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

build_dir = Path(__file__).parent.parent.resolve()

"""Tests of the example PopART cube custom operator"""


@pytest.fixture(autouse=True)
def with_compiled_op():
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("make", build_dir, [])


@pytest.mark.ipus(1)
@pytest.mark.category1
def test_run_ipu():
    "Run test on an IPU device of the custom cube operator on a input vector."
    "Validate that the output and output gradients are correct"
    testing_util.run_command(
        "./build/cube_fn_custom_op --ipu 0.3 2.7 1.2 5",
        build_dir,
        [
            "Running the example on IPU.",
            "Output Data: {0.027, 19.683, 1.728, 125}",
            "Input Grad:  {0.00081, 0.59049, 0.05184, 3.75}",
        ],
    )


@pytest.mark.category1
def test_run_ipu_model():
    "Run test on a CPU device of the custom cube operator on a input vector."
    "Validate that the output and output gradients are correct"
    testing_util.run_command(
        "./build/cube_fn_custom_op 0.3 2.7 1.2 5",
        build_dir,
        [
            "Running the example on CPU.",
            "Output Data: {0.027, 19.683, 1.728, 125}",
            "Input Grad:  {0.00081, 0.59049, 0.05184, 3.75}",
        ],
    )


if __name__ == "__main__":
    pytest.main(args=[__file__, "-s"])
