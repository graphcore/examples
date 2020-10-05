# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


class TestBuildAndRun(SubProcessChecker):

    def _run_test_matmul(self, cmd_args, extra_env={}):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        env.update(extra_env)
        self.run_command(cmd_args, build_dir, ["Results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_sparse_matmul_default(self):
        # Test default parameters:
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_random_pattern(self):
        # Test the random pattern option:
        self._run_test_matmul("python ipu_sparse_ops/tools/sparse_matmul.py --batch-size 16 --input-size 64 --output-size 32 --pattern random --density 0.05")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_mnist_fc_size_matmul(self):
        # Test an MNIST FC sized matmul:
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 784 --output-size 300 --batch-size 16")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_larger_size(self):
        # Test a larger size:
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 8192 --output-size 1536 --batch-size 8",
                              extra_env={'POPLAR_ENGINE_OPTIONS': '{"opt.internalExchangeOptimisationTarget": "balanced"}'})

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_half_precision_small(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --data-type=fp16 --batch-size 4")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_half_precision_mnist_size(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --data-type=fp16 --input-size 784 --output-size 300 --batch-size 16")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_random_sign_ones(self):
        # Test random sign ones (with zero tolerance):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 8192 --output-size 1536 --batch-size 6 --pattern random_sign_ones",
                              extra_env={'POPLAR_ENGINE_OPTIONS': '{"opt.internalExchangeOptimisationTarget": "balanced"}'})

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_random_orthogonal(self):
        # Test random orthogonal initialiser (with smaller tolerance):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 1920 --output-size 1920 --batch-size 2 --pattern random_orthogonal")
