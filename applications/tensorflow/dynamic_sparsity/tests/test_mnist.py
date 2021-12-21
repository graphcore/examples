# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import sys
import pytest

build_dir = Path(__file__).parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestBuildAndRun(SubProcessChecker):

    def _run_command(self, args=""):
        cmd = sys.executable + " mnist_rigl/sparse_mnist.py"
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        self.run_command(f"{cmd} {args}",
                         build_dir,
                         [r"Test loss: (\w+.\w+) Test accuracy: (\w+.\w+)"], env=env)

    @pytest.mark.ipus(1)
    def test_sparse_mnist_default(self):
        self._run_command()

    @pytest.mark.ipus(1)
    def test_sparse_mnist_adam(self):
        self._run_command("--optimizer Adam")

    @pytest.mark.ipus(1)
    def test_sparse_mnist_mixed(self):
        self._run_command("--densities 0.1 0.2")

    @pytest.mark.ipus(1)
    def test_sparse_mnist_mixed_blocks_fp16(self):
        self._run_command("--densities 0.1 0.2 --block-size 16 --data-type fp16 --partials-type fp16")

    @pytest.mark.ipus(1)
    def test_sparse_mnist_blocks_random_regrow(self):
        self._run_command("--block-size 16 --regrow random")

    @pytest.mark.ipus(1)
    def test_sparse_mnist_pipelined(self):
        self._run_command("--densities 0.1 1 --pipelining --gradient-accumulation-count 4 --block-size 1")

    @pytest.mark.ipus(1)
    def test_sparse_mnist_pipelined_blocks(self):
        self._run_command("--densities 0.1 1 --pipelining --gradient-accumulation-count 4 --block-size 16")
