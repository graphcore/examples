# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pathlib
import os
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

working_path = pathlib.Path(__file__).parents[1]

### Pytest markers
@pytest.mark.ipus("<nb of IPUs required>")
def test_notebook_name(tmp_path):
    """
    <Describe test purpose>
    """
    os.environ["NUM_AVAILABLE_IPU"] = "4"
    notebook_filename = working_path / "notebook.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{working_path}"}})
