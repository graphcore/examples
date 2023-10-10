# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pathlib
import re

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


EXAMPLE_ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()


def test_inference_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "gps++_inference.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})


def test_training_notebook():
    notebook_filename = EXAMPLE_ROOT_DIR / "gps++_training.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    gen_data_replacement = (
        "cfg.dataset.generated_data_n_graphs = 10000\n" "cfg.dataset.dataset_name = 'generated_extended'"
    )

    print("Attempting to reduce number of epochs used in notebook to 1.")
    print("Attempting to change dataset type to generated.")
    for cell in nb.cells:
        # Reduce number of epochs run by notebook to 1
        cell["source"] = re.sub(r"cfg.model.epochs = (\d+)", "cfg.model.epochs = 1", cell["source"])
        # Change to use generated data
        cell["source"] = re.sub(
            r"print\(f\"Dataset: {cfg.dataset.dataset_name}\"\)", gen_data_replacement, cell["source"]
        )

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{EXAMPLE_ROOT_DIR}"}})
