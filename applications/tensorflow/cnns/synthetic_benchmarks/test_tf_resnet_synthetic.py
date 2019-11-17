# Copyright 2019 Graphcore Ltd.
import inspect
import unittest
import os
import sys
import subprocess
from contextlib import contextmanager

import tests.test_util as tu


def run_resnet(**kwargs):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = ["python" + str(sys.version_info[0]), 'resnet.py']
    # Flatten kwargs and convert to strings
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
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


if __name__ == '__main__':
    unittest.main()
