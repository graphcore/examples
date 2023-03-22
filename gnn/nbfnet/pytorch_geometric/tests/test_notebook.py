# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pathlib
import re

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


EXAMPLE_ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()


def test_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "NBFNet_training.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    # Reduce number of epochs run by notebook to 1
    print("Attempting to reduce number of epochs used in notebook to 1.")
    for cell in nb.cells:
        cell["source"] = re.sub(r"num_epochs = (\d)", "num_epochs = 1", cell["source"])

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})
