# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os
import pathlib
import re

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


EXAMPLE_ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()


def test_dataset_preprocessing_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "1_dataset_preprocessing.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    dataset_directory = pathlib.Path(__file__).parent.joinpath("dummy_dataset").resolve()
    os.environ["DATASETS_DIR"] = str(dataset_directory)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})


def test_training_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "2_training.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    dataset_directory = pathlib.Path(__file__).parent.joinpath("dummy_dataset").resolve()
    os.environ["DATASETS_DIR"] = str(dataset_directory)

    # Reduce number of epochs run by notebook to 1
    print("Attempting to reduce number of epochs used in notebook to 1.")
    for cell in nb.cells:
        cell["source"] = re.sub(r"num_epochs = (\d)", "num_epochs = 1", cell["source"])

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})
