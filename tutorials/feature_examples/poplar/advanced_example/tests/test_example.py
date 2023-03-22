# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path

import pytest
from filelock import FileLock

# NOTE: The imports below are dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

build_dir = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def setUp():
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("make", build_dir, [])


@pytest.mark.category1
def test_run_ipu_model():
    testing_util.run_command("./example --model", build_dir, ["Results match.", "Using IPU model"])


@pytest.mark.ipus(1)
@pytest.mark.category1
def test_run_ipu():
    testing_util.run_command("./example", build_dir, ["Using HW device ID", "Results match."])
