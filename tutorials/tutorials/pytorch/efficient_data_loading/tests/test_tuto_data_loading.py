# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path

import pytest
from tutorials_tests import testing_util
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

working_path = Path(__file__).parent.parent


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_run_default_ipu():
    # Check default params
    testing_util.run_command("python tuto_data_loading.py", working_path, "IPU throughput")


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_run_synthetic_ipu():
    # Check synthetic data params
    testing_util.run_command(
        "python tuto_data_loading.py --synthetic-data",
        working_path,
        "IPU throughput",
    )


@pytest.mark.category2
@pytest.mark.ipus(2)
def test_run_replication_ipu():
    # Check replication
    testing_util.run_command("python tuto_data_loading.py --replicas 2", working_path, "IPU throughput")


@pytest.mark.category1
@pytest.mark.ipus(2)
def test_run_replication_synthetic_ipu():
    # Check synthetic data with replication
    testing_util.run_command(
        "python tuto_data_loading.py --replicas 2 --synthetic-data",
        working_path,
        "IPU throughput",
    )


@pytest.mark.ipus(4)
@pytest.mark.category1
def test_notebook():
    notebook_filename = working_path / "walkthrough.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{working_path}"}})
