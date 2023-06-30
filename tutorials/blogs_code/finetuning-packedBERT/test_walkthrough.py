# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path
from examples_utils.benchmarks.notebook_utils import run_notebook

example_dir = Path(__file__).parent


def test_smoke(tmp_path):
    run_notebook(example_dir / "walkthrough.ipynb", tmp_path)
