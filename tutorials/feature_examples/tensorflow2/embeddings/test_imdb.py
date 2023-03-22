# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import pytest
from filelock import FileLock
import tutorials_tests.testing_util as testing_util

working_path = os.path.dirname(__file__)


"""Integration tests for TensorFlow 2 IMDB example"""


@pytest.fixture(autouse=True, scope="module")
def download_imdb():
    # We import after the start of the session to allow the tests
    # to be discovered without requiring test specific dependencies.
    from imdb import get_dataset

    with FileLock("download_imdb.lock"):
        get_dataset()


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_pipeline():
    testing_util.run_command("python imdb.py", working_path, "Epoch 2/")


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_pipeline_sequential():
    testing_util.run_command("python imdb_sequential.py", working_path, "Epoch 2/")


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_single_ipu():
    testing_util.run_command("python imdb_single_ipu.py", working_path, "Epoch 3/")


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_single_ipu_sequential():
    testing_util.run_command("python imdb_single_ipu_sequential.py", working_path, "Epoch 3/")
