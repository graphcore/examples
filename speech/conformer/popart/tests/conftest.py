# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import os
from pathlib import Path
from subprocess import run
from test_utils import load_custom_ops_lib
import sys

# Add the application root to the PYTHONPATH
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)


@pytest.fixture
def custom_ops():
    return load_custom_ops_lib()


def pytest_sessionstart(session):
    """ this method will be called after the pytest session has been created and before running the test loop """
    pass
