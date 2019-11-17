#!/usr/bin/python
# Copyright 2019 Graphcore Ltd.

import inspect
import os
import pytest
import sys
import unittest

import tests.test_util as test_util
from pytorch_popart_mnist import get_data_loader


def run_pytorch_mnist(**kwargs):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    return test_util.run_python_script_helper(
        cwd, "pytorch_popart_mnist.py", **kwargs
    )


class TestPytorchMNISTTraining(unittest.TestCase):
    """High-level integration tests for the PyTorch model training on
       the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        cls.time_tolerances = 0.2
        cls.generic_arguments = {
            "--batch-size": 32,
            "--batches-per-step": 100,
            "--epochs": 10
        }

    def test_log_graph_trace_arg(self):
        """Generic test exercising log graph trace argument"""
        py_args = self.generic_arguments.copy()
        py_args["--log-graph-trace"] = ""
        py_args["--epochs"] = 1
        test_util.run_test_helper(
            run_pytorch_mnist,
            **py_args
        )

    def test_train_32_100_1_simulation(self):
        """Generic test in simulation mode"""
        py_args = self.generic_arguments.copy()
        py_args["--simulation"] = ""
        py_args["--epochs"] = 1
        test_util.run_test_helper(
            run_pytorch_mnist,
            **py_args
        )

    def test_train_32_100_10(self):
        """Generic functional test"""
        py_args = self.generic_arguments.copy()
        out = test_util.run_test_helper(
            run_pytorch_mnist,
            **py_args
        )
        expected_accuracy = [
            87.65, 88.06, 88.4, 88.43, 88.68, 88.71, 88.69, 88.89, 88.85, 88.61
        ]
        test_util.parse_results_for_accuracy(out, expected_accuracy, 2.5)


if __name__ == "__main__":
    unittest.main()
