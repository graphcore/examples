# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import pytest
import gc
import import_helper
from utils import get_cifar10_dataset, get_models
from examples_tests.execute_once_per_fs import ExecuteOncePerFS


@pytest.fixture(autouse=True)
def cleanup():
    # Explicitly clean up to make sure we detach from the IPU and
    # free the graph before the next test starts.
    gc.collect()


@ExecuteOncePerFS(lockfile="lockfile_serial.lock", file_list=[], timeout=120, retries=20)
def init_tests():
    """Get the data required for the tests."""
    get_cifar10_dataset()
    get_models()


def pytest_sessionstart(session):
    init_tests()
    os.environ["POPTORCH_WAIT_FOR_IPU"] = "1"
