# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestBuildAndRun(SubProcessChecker):

    def _run_test_matmul(self, cmd_args, extra_env={}):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        env.update(extra_env)
        self.run_command(cmd_args, build_dir, ["Results match."], env=env)

    @pytest.mark.ipus(1)
    def test_sparse_matmul_default(self):
        # Test default parameters:
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py")

    @pytest.mark.ipus(1)
    def test_random_pattern(self):
        # Test the random pattern option:
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --batch-size 16 --input-size 64 --output-size 32 --pattern random --density 0.05")

    @pytest.mark.ipus(1)
    def test_mnist_fc_size_matmul(self):
        # Test an MNIST FC sized matmul:
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 784 --output-size 300 --batch-size 16")

    @pytest.mark.ipus(1)
    def test_larger_size(self):
        # Test a larger size:
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 8192 --output-size 1536 --batch-size 8",
                              extra_env={'POPLAR_ENGINE_OPTIONS': '{"opt.internalExchangeOptimisationTarget": "balanced"}'})

    @pytest.mark.ipus(1)
    def test_half_precision_small(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --data-type=fp16 --batch-size 4")

    @pytest.mark.ipus(1)
    def test_half_precision_mnist_size(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --data-type=fp16 --input-size 784 --output-size 300 --batch-size 16")

    @pytest.mark.ipus(1)
    def test_random_sign_ones(self):
        # Test random sign ones (with zero tolerance):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 8192 --output-size 1536 --batch-size 6 --pattern random_sign_ones",
                              extra_env={'POPLAR_ENGINE_OPTIONS': '{"opt.internalExchangeOptimisationTarget": "balanced"}'})

    @pytest.mark.ipus(1)
    def test_random_orthogonal(self):
        # Test random orthogonal initialiser (with smaller tolerance):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 1920 --output-size 1920 --batch-size 2 --pattern random_orthogonal")

    @pytest.mark.ipus(1)
    def test_fixed_sweep(self):
        for dtype in ['fp16', 'fp32']:
            for b in [1, 4, 8, 16]:
                self._run_test_matmul(f"python3 ipu_sparse_ops/tools/sparse_matmul.py --batch-size=4 --input-size {3*b} --output-size {8*b} --data-type {dtype} --block-size {b} --pattern=fixed")

    @pytest.mark.ipus(1)
    def test_random_16x16(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --batch-size=1 --input-size 64 --output-size 128 --data-type fp32 --block-size 16 --pattern=random")

    @pytest.mark.ipus(1)
    def test_large_random_16x16_fp16(self):
        # Test random sign ones (with zero tolerance):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --density 0.1 --input-size 8192 --output-size 1536 --batch-size 512 --pattern random --data-type fp16 --block-size 16")

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_large_random_sign_ones_16x16_fp16(self):
        # Test random sign ones (with zero tolerance):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --density 0.2 --input-size 4096 --output-size 1536 --batch-size 512 --pattern random_sign_ones --data-type fp16 --block-size 16")

    @pytest.mark.ipus(1)
    def test_mnist_fc_size_16x16(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --batch-size=1 --input-size 784 --output-size 320 --data-type fp32 --block-size 16 --pattern=random")

    @pytest.mark.ipus(1)
    def test_mnist_fc_size_8x8(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --batch-size=1 --input-size 784 --output-size 320 --data-type fp32 --block-size 8 --pattern=random")

    @pytest.mark.ipus(1)
    def test_mnist_fc_size_4x4(self):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --batch-size=1 --input-size 784 --output-size 320 --data-type fp32 --block-size 4 --pattern=random")

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_gpt2_boom_16x16_fp16(self):
        # Test random sign ones (with zero tolerance):
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 1600 --output-size 6400 --batch-size 16 --pattern random_sign_ones --data-type fp16 --block-size 16")
        self._run_test_matmul("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 6400 --output-size 1600 --batch-size 16 --pattern random_sign_ones --data-type fp16 --block-size 16")
