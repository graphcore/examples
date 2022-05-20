# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Simple tests for the DeepDriveMD CVAE"""
import pytest
from pathlib import Path
from examples_tests.test_util import SubProcessChecker


current_path = Path(__file__).parent


class DeepDriveMDTest(SubProcessChecker):
    """Test simple command line executions"""

    @pytest.mark.ipus(1)
    def test_default(self):
        """Test the defaults"""
        self.run_command(
            "python train_cvae.py",
            current_path,
            ["Average throughput"])

    @pytest.mark.ipus(1)
    def test_args(self):
        """Test the args"""
        self.run_command(
            "python train_cvae.py --batch_size 200 --num_epochs 5 --img_size 20 --dataset_size 8192 --no-validation",
            current_path,
            ["Average throughput"])
