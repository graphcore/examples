# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestTensorFlowGroupedConvBenchmarks(SubProcessChecker):
    """High-level integration tests for TensorFlow grouped convolution synthetic benchmarks"""

    @pytest.mark.category1
    def test_help(self):
        self.run_command("python3 grouped_conv.py --help",
                         working_path,
                         "usage: grouped_conv.py")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_default(self):
        self.run_command("python3 grouped_conv.py",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_inference(self):
        self.run_command("python3 grouped_conv.py --batch-size 8 --use-generated-data",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_block_repeats_and_group_dims(self):
        self.run_command("python3 grouped_conv.py --block-repeats 20 --group-dim 8",
                         working_path,
                         [r"(\w+.\w+) items/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_training(self):
        self.run_command("python3 grouped_conv.py --train --input-size 112  --stride 3 --filter-in 32 --filter-out 16",
                         working_path,
                         [r"(\w+.\w+) items/sec", "Input size 112"])

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_replicas(self):
        self.run_command("python3 grouped_conv.py --replicas 2",
                         working_path,
                         [r"(\w+.\w+) items/sec"])
