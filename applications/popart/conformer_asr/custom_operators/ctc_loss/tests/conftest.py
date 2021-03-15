# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import os
from pathlib import Path
from subprocess import run

from test_utils import load_custom_ops_lib

custom_ops_loaded = False


@pytest.fixture
def custom_ops():
    if not custom_ops_loaded:
        return load_custom_ops_lib()
    else:
        return "ops loaded"


def rebuild_custom_ops():
    """ The objective of this method is to:
    1.) Delete the existing CTC loss op if it exists
    2.) Perform the make command
    3.) Validate a ctc_loss.so does exist """
    model_path = Path(__file__).resolve().parent.parent
    custom_ops_path = Path(model_path, "build/ctc_loss.so")
    if custom_ops_path.exists():
        print(f"\nDeleting: {custom_ops_path}")
        os.remove(custom_ops_path)
    print("\nBuilding CTC Custom Loss Op")
    run(["make"], cwd=custom_ops_path.parent.parent)
    assert custom_ops_path.exists()


def pytest_sessionstart(session):
    """ this method will be called after the pytest session has been created and before running the test loop """
    rebuild_custom_ops()
