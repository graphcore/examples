# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pathlib
from utils import build_custom_ops
from mnist.common import download_mnist
import pytest
import ctypes
import os


def pytest_sessionstart(session):
    # Builds the custom ops
    so_path = pathlib.Path(__file__).parent.parent.resolve() / "custom_ops.so"
    build_custom_ops(so_path)

    # Download MNIST dataset
    download_mnist(pathlib.Path(__file__).parent.resolve() / "mnist")

    # Sets the IPUs to wait before attaching.
    os.environ["POPTORCH_WAIT_FOR_IPU"] = "1"


@pytest.fixture
def custom_ops():
    so_path = pathlib.Path(__file__).parent.parent.resolve() / "custom_ops.so"
    ctypes.cdll.LoadLibrary(so_path)
    return so_path
