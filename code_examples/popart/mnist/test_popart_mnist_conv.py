# Copyright 2019 Graphcore Ltd.
import inspect
import os
import subprocess
import sys
import unittest

from test_popart_mnist import download_mnist
import tests.test_util as test_util


def run_popart_mnist_training(**kwargs):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    out = test_util.run_python_script_helper(
        cwd, "popart_mnist_conv.py", **kwargs
    )
    return out


class TestPopARTMNISTImageClassificationConvolution(unittest.TestCase):
    """High-level integration tests for training with the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        download_mnist(cwd)
        cls.accuracy_tolerances = 3.0
        cls.generic_arguments = {
            "--batch-size": 4,
            "--batches-per-step": 1000,
            "--epochs": 10
        }

    def test_mnist_train(self):
        """Generic test on default arguments in training"""
        py_args = self.generic_arguments.copy()
        out = test_util.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [
            97.72, 98.15, 98.51, 98.55, 98.55, 98.38, 98.34, 98.35, 98.43, 98.41
        ]
        test_util.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    def test_mnist_all_data(self):
        """Generic test using all the available data (10,000)"""
        py_args = self.generic_arguments.copy()
        py_args["--epochs"] = 2
        py_args["--batch-size"] = 10
        py_args["--batches-per-step"] = 1000
        test_util.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )

    def test_mnist_log_graph_trace(self):
        """Basic test with log-graph-trace argument"""
        py_args = self.generic_arguments.copy()
        py_args["--log-graph-trace"] = ""
        py_args["--epochs"] = 1
        test_util.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )

    def test_mnist_conv_simulation(self):
        """Simulation test with basic arguments - This simulation takes
           around 838s (~14 minutes) complete"""
        py_args = self.generic_arguments.copy()
        py_args["--simulation"] = ""
        py_args["--batch-size"] = 1
        py_args["--batches-per-step"] = 1
        py_args["--epochs"] = 1
        test_util.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )


if __name__ == "__main__":
    unittest.main()
