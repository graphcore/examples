# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pathlib
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from examples_utils.testing.test_commands import run_command_fail_explicitly


EXAMPLE_ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()


def test_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "molecular_property_prediction_with_schnet.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})


def test_python_file_code_only():
    python_filename = EXAMPLE_ROOT_DIR / "molecular_property_prediction_with_schnet_code_only.py"
    run_command_fail_explicitly([sys.executable, python_filename], cwd=EXAMPLE_ROOT_DIR)
