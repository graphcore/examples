# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import sys
import pathlib
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from tutorials_tests import testing_util

TUTORIAL_ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()


@pytest.mark.ipus(2)
@pytest.mark.category1
def test_notebook():
    notebook_filename = TUTORIAL_ROOT_DIR / "mnist.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{TUTORIAL_ROOT_DIR}"}})


@pytest.mark.ipus(4)
@pytest.mark.category1
def test_notebook_rg():
    notebook_filename = TUTORIAL_ROOT_DIR / "replica_groupings.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{TUTORIAL_ROOT_DIR}"}})


@pytest.mark.ipus(2)
@pytest.mark.category1
def test_python_file():
    python_filename = TUTORIAL_ROOT_DIR / "mnist.py"
    testing_util.run_command_fail_explicitly([sys.executable, python_filename], cwd=TUTORIAL_ROOT_DIR)
