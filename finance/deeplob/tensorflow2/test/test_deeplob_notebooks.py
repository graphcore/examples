# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pathlib
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from examples_utils.testing.test_commands import run_command_fail_explicitly


EXAMPLE_ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()


def test_seq2seq_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "Multi-Horizon-DeepLOBSeq2Seq-training.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})


def test_attention_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "Multi-Horizon-DeepLOBAttention-training.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})
