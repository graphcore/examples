# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Tests for the EfficientDet inference implementation."""
from pathlib import Path
import pytest
from examples_tests.test_util import SubProcessChecker

current_path = Path(__file__).parent.parent


class ThroughputTest(SubProcessChecker):
    """Test throughput examples using ipu_inference.py."""

    def setUp(self):
        self.run_command("make", current_path, [])

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_d0_bs1(self):
        """Test throughput inference on EfficientDet D0"""
        self.run_command(
            "python ipu_inference.py --model-name efficientdet-d0 --micro-batch-size 1 --random-weights", current_path, ["Benchmark complete"])

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_d0_throughput(self):
        """Test throughput inference on EfficientDet D0 BS4"""
        self.run_command(
            "python ipu_inference.py --model-name efficientdet-d0 --random-weights", current_path, ["Benchmark complete"])

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_d4_throughput(self):
        """Test throughput inference on EfficientDet D4 BS 1"""
        self.run_command(
            "python ipu_inference.py --model-name efficientdet-d4 --random-weights", current_path, ["Benchmark complete"])


class LowLatencyTest(SubProcessChecker):
    """Test low-latency benchmark using ipu_embedded_inference.py."""

    def setUp(self):
        self.run_command("make", current_path, [])

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_d0_bs1(self):
        """Test low latency single image inference on EfficientDet D0"""
        self.run_command(
            "python ipu_embedded_inference.py --model-name efficientdet-d0 --config efficientdet-low-latency", current_path, ["Benchmark complete"])

    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_d4_bs1(self):
        """Test low latency single image inference on EfficientDet D4"""
        self.run_command(
            "python ipu_embedded_inference.py --model-name efficientdet-d4 --config efficientdet-low-latency", current_path, ["Benchmark complete"])
