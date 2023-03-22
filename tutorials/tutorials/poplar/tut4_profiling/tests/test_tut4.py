# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import tempfile
from pathlib import Path
from shutil import copy

import pytest
from filelock import FileLock

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def with_compiled_examples():
    """Compile the start here and complete versions of the tutorial code"""
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("make all", working_path)


@pytest.mark.category1
def test_run_ipu_hardware():
    """Check that the hardware version of the tutorial code runs"""
    testing_util.run_command("./tut4_ipu_hardware", working_path, ["Program complete", "Memory Usage:"])


@pytest.mark.category1
def test_run_ipu_model():
    """Check that the IPUModel version of the tutorial code runs"""
    testing_util.run_command("./tut4_ipu_model", working_path, ["Program complete", "Memory Usage:"])


@pytest.mark.category1
def test_run_cpp_example():
    """Check that the cpp_example can open a profile.pop report"""

    # Set environment var to collect reports
    env = os.environ.copy()
    env["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all": "true"}'

    with tempfile.TemporaryDirectory() as temporary_path:
        # Copy binaries to temp directory
        copy(working_path / "tut4_ipu_model", Path(temporary_path) / "tut4_ipu_model")
        copy(working_path / "cpp_example", Path(temporary_path) / "cpp_example")

        # Execute ipu_model that will collect reports
        testing_util.run_command(
            "./tut4_ipu_model",
            temporary_path,
            ["Program complete", "Memory Usage:"],
            env=env,
        )

        # Inspect reports
        testing_util.run_command("./cpp_example", temporary_path, ["Example information from profile"])
