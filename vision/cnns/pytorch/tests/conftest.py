# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path
import json
import pytest
import gc
import torchvision
from torch.utils.data import DataLoader
import import_helper
from utils import download_images, get_models, get_cifar10_dataset, install_turbo_jpeg
from examples_tests.execute_once_per_fs import ExecuteOncePerFS


@pytest.fixture(autouse=True)
def cleanup():
    # Explicitly clean up to make sure we detach from the IPU and
    # free the graph before the next test starts.
    gc.collect()


@ExecuteOncePerFS(lockfile="lockfile.lock", file_list=[], timeout=120, retries=20)
def init_tests():
    """Get the data required for the tests."""
    get_cifar10_dataset()
    download_images()
    get_models()
    install_turbo_jpeg()


def pytest_sessionstart(session):
    init_tests()
    os.environ["POPTORCH_WAIT_FOR_IPU"] = "1"
