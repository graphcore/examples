# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import unittest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, run_test_helper, \
    parse_results_for_accuracy


def run_pytorch_mnist(**kwargs):
    return run_python_script_helper(
        os.path.dirname(__file__), "pytorch_popart_mnist.py", **kwargs
    )


class TestPytorchMNISTTraining(unittest.TestCase):
    """High-level integration tests for the PyTorch model training on
       the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cls.time_tolerances = 0.2
        cls.generic_arguments = {
            "--batch-size": 32,
            "--batches-per-step": 100,
            "--epochs": 10,
            "--validation-final-epoch": "",
        }

    def test_log_graph_trace_arg(self):
        """Generic test exercising log graph trace argument"""
        py_args = self.generic_arguments.copy()
        py_args["--log-graph-trace"] = ""
        py_args["--epochs"] = 1
        run_test_helper(
            run_pytorch_mnist,
            **py_args
        )

    def test_train_32_100_1_simulation(self):
        """Generic test in simulation mode"""
        py_args = self.generic_arguments.copy()
        py_args["--simulation"] = ""
        py_args["--epochs"] = 1
        run_test_helper(
            run_pytorch_mnist,
            **py_args
        )

    def test_train_32_100_10(self):
        """Generic functional test"""
        py_args = self.generic_arguments.copy()
        out = run_test_helper(
            run_pytorch_mnist,
            **py_args
        )
        expected_accuracy = [88.61]
        parse_results_for_accuracy(out, expected_accuracy, 2.5)
