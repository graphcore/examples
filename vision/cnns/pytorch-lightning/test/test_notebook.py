# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pathlib
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

EXAMPLE_ROOT_DIR = pathlib.Path(__file__).parent.parent / "code-examples" / "fashion-mnist"


@pytest.mark.category2
@pytest.mark.ipus(4)
def test_notebook_1():
    notebook_filename = EXAMPLE_ROOT_DIR / "fashionmnist.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})


@pytest.mark.category2
@pytest.mark.ipus(4)
def test_notebook_2():
    notebook_filename = EXAMPLE_ROOT_DIR / "fashionmnist_torchvision.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})
