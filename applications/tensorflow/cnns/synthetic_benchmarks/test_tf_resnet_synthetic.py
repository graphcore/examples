# Copyright 2020 Graphcore Ltd.
import os
import subprocess
import sys
import unittest


def run_resnet(**kwargs):
    cmd = ["python" + str(sys.version_info[0]), 'resnet.py']
    # Flatten kwargs and convert to strings
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    cwd = os.path.dirname(__file__)
    test_env = os.environ.copy()
    test_env["LANG"] = "C.UTF-8"
    out = subprocess.check_output(cmd, cwd=cwd, env=test_env).decode("utf-8")
    print(out)
    return out


class TestTensorFlowResNetSyntheticBenchmarks(unittest.TestCase):
    """High-level integration tests for TensorFlow CNN synthetic benchmarks"""

    @classmethod
    def setUpClass(cls):
        pass

    # Resnet inference
    def test_resnet_18_inference_batch_size_1(self):
        out = run_resnet(**{'--size': 18, '--batch-size': 1})

    def test_resnet_18_inference_batch_size_16(self):
        out = run_resnet(**{'--size': 18, '--batch-size': 16})

    def test_resnet_50_inference_batch_size_1(self):
        out = run_resnet(**{'--size': 50, '--batch-size': 1})

    def test_resnet_50_inference_batch_size_8(self):
        out = run_resnet(**{'--size': 50, '--batch-size': 8})
