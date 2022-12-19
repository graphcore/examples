# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""Tests for the UNet implementation."""
from pathlib import Path
import pytest
from examples_tests.test_util import SubProcessChecker

current_path = Path(__file__).parent


class UNetTest(SubProcessChecker):

    def setUp(self):
        self.run_command("make", current_path, [])

    """Test training command line executions."""
    @pytest.mark.ipus(4)
    def test_train(self):
        """Test the training on 4 IPUs"""
        self.run_command(
            "python3 main.py --nb-ipus-per-replica 4 --micro-batch-size 1 --gradient-accumulation-count 24 --num-epochs 20 --train --augment --learning-rate 0.0001 --host-generated-data", current_path, ["Training complete"])


    """Test inference command line executions."""
    @pytest.mark.ipus(4)
    def test_infer(self):
        """Test the inference on 4 IPUs"""
        self.run_command(
            "python3 main.py --nb-ipus-per-replica 1 --replicas 4 --micro-batch-size 2 --gradient-accumulation-count 1 --infer --host-generated-data --benchmark", current_path, ["Inference complete"])
