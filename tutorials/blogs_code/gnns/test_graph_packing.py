# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys
import pytest
from pathlib import Path

import nbformat as nbf

from examples_utils.benchmarks.notebook_utils import run_notebook
from examples_utils.testing.test_commands import run_command_fail_explicitly

example_dir = Path(__file__).parent


def generate_test_file(root_path):
    """The full notebook takes too long to run to include in tests. Instead we'll export the first few
    cells to test the packing algorithm and dataset setup only.

    We'll also modify the list of datasets for this test so as to only test on the smallest dataset.
    """

    nbtk = nbf.read(example_dir / "graph_packing.ipynb", nbf.NO_CONVERT)
    code_cells = [c["source"] for c in nbtk.cells if c["cell_type"] == "code"]

    search_term = '["ogbg-molhiv", "ogbg-molpcba", "ogbg-code2", "ogbg-pcqm4mv2", "ogbg-ppa"]'
    replacement = '["ogbg-molhiv"]'

    fixed_cell = code_cells[1].replace(search_term, replacement)

    if fixed_cell == code_cells[1]:
        raise Exception("Could not replace datasets for the test. Has the notebook changed?")

    code_cells[1] = fixed_cell

    output_path = root_path / "converted_graph_packing.py"

    with open(output_path, "w") as fh:
        for cell in code_cells[:9]:
            fh.write(cell + "\n\n")

    return output_path


def test_smoke(tmp_path):
    script_path = generate_test_file(tmp_path)
    run_command_fail_explicitly([sys.executable, script_path.name], cwd=script_path.parent)
