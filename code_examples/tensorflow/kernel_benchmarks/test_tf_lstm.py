# Copyright 2019 Graphcore Ltd.
import inspect
import unittest
import os
import sys
import subprocess
from contextlib import contextmanager

import tests.test_util as tu


def run_lstm(batch_size, timesteps, hidden_size, extra_args=None):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    py_version = "python" + str(sys.version_info[0])
    cmd = [py_version, "lstm.py",
           "--batch-size", str(batch_size),
           "--timesteps", str(timesteps),
           "--hidden-size", str(hidden_size)]
    # Accommodate framework-specific args
    if extra_args:
        cmd.extend(extra_args)
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
    print(out)
    return out


class TestTensorFlowLSTMBenchmarks(unittest.TestCase):
    """High-level integration tests for TensorFlow LSTM synthetic benchmarks"""

    @classmethod
    def setUpClass(cls):
        pass

    # LSTM inference
    def test_lstm_inference_b1024_s25_h512(self):
        out = run_lstm(batch_size=256, timesteps=25, hidden_size=1024, extra_args=["--popnn"])

    def test_lstm_inference_b128_s50_h1536(self):
        out = run_lstm(batch_size=128, timesteps=50, hidden_size=1536, extra_args=["--popnn"])

    def test_lstm_inference_b64_s25_h2048(self):
        out = run_lstm(batch_size=64, timesteps=25, hidden_size=2048, extra_args=["--popnn"])

    def test_lstm_inference_b1024_s150_h256(self):
        out = run_lstm(batch_size=1024, timesteps=150, hidden_size=256, extra_args=["--popnn"])

    def test_lstm_inference_b1024_s25_h512(self):
        out = run_lstm(batch_size=1024, timesteps=25, hidden_size=512, extra_args=["--popnn"])


if __name__ == '__main__':
    unittest.main()
