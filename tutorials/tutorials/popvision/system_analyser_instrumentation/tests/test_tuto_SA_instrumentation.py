# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from pathlib import Path

import pytest
from filelock import FileLock
from tutorials_tests import testing_util
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from tutorials_tests import testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category2
def test_tutorial_code():
    expected_strings = ["Graph compilation:", "batches:", "epochs:", "Epoch #1", "Loss="]
    with FileLock(__file__ + ".lock"):
        testing_util.run_command("python3 walkthrough_code_only.py", working_path, expected_strings)


@pytest.mark.ipus(2)
@pytest.mark.category1
def test_notebook():
    notebook_filename = working_path / "walkthrough.ipynb"
    with FileLock(__file__ + ".lock"):
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": f"{working_path}"}})
