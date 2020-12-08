# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
import os

import pytest
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestTensorFlowSparseFcBenchmarks(SubProcessChecker):
    """High-level integration tests for TensorFlow dynamic sparse FC layer benchmarks"""

    @pytest.mark.category1
    def test_help(self):
        self.run_command("python3 dynamic_sparse_fc.py --help",
                         working_path,
                         "usage: dynamic_sparse_fc.py")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_default(self):
        self.run_command("python3 dynamic_sparse_fc.py",
                         working_path,
                         [r"(\w+.\w+) items/sec (\w+.\w+) TFLOPS/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_16x16_blocks(self):
        self.run_command("python3 dynamic_sparse_fc.py --block-size 16 --batch-size=1 --input-size=160 --output-size=160",
                         working_path,
                         [r"(\w+.\w+) items/sec (\w+.\w+) TFLOPS/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_default_fp16(self):
        self.run_command("python3 dynamic_sparse_fc.py --data-type fp16",
                         working_path,
                         [r"(\w+.\w+) items/sec (\w+.\w+) TFLOPS/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_all_args(self):
        self.run_command("python3 dynamic_sparse_fc.py --input-size 512 --output-size 1024 --batch-size 16 --batches-per-step 10 --density 0.2",
                         working_path,
                         [r"(\w+.\w+) items/sec (\w+.\w+) TFLOPS/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_train(self):
        self.run_command("python3 dynamic_sparse_fc.py --mode train",
                         working_path,
                         [r"(\w+.\w+) items/sec (\w+.\w+) TFLOPS/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_train_real_data(self):
        self.run_command("python3 dynamic_sparse_fc.py --mode train --use-generated-data",
                         working_path,
                         [r"(\w+.\w+) items/sec (\w+.\w+) TFLOPS/sec"])
