# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import sys
import pytest

build_dir = Path(__file__).parent.parent


class TestBuildAndRun(SubProcessChecker):

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_sparse_mnist(self):
        cmd = sys.executable
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        # Test default parameters:
        self.run_command(f"{cmd} mnist_rigl/sparse_mnist.py",
                         build_dir,
                         [r"Test loss: (\w+.\w+) Test accuracy: (\w+.\w+)"], env=env)
        # Test Adam optimizer:
        self.run_command(f"{cmd} mnist_rigl/sparse_mnist.py --optimizer Adam",
                         build_dir,
                         [r"Test loss: (\w+.\w+) Test accuracy: (\w+.\w+)"], env=env)
        # Test both layers sparse as in the README:
        self.run_command(f"{cmd} mnist_rigl/sparse_mnist.py --densities 0.1 0.2",
                         build_dir,
                         [r"Test loss: (\w+.\w+) Test accuracy: (\w+.\w+)"], env=env)
