# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import unittest

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests import test_util


def run_popart_mnist_training(**kwargs):
    out = test_util.run_python_script_helper(
        os.path.dirname(__file__), "popart_mnist_conv.py", **kwargs
    )
    return out


class TestPopARTMNISTImageClassificationConvolution(unittest.TestCase):
    """High-level integration tests for training with the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cls.accuracy_tolerances = 3.0
        cls.generic_arguments = {
            "--batch-size": 4,
            "--batches-per-step": 1000,
            "--epochs": 10,
            "--validation-final-epoch": "",
        }

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_train(self):
        """Generic test on default arguments in training"""
        py_args = self.generic_arguments.copy()
        out = test_util.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [98.41]
        test_util.parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(1)
    @pytest.mark.category2
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

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_log_graph_trace(self):
        """Basic test with log-graph-trace argument"""
        py_args = self.generic_arguments.copy()
        py_args["--log-graph-trace"] = ""
        py_args["--epochs"] = 1
        test_util.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )

    @pytest.mark.category3
    def test_mnist_conv_simulation(self):
        """Simulation test with basic arguments - This simulation takes
           around 838s (~14 minutes) to complete"""
        py_args = self.generic_arguments.copy()
        py_args["--simulation"] = ""
        py_args["--batch-size"] = 1
        py_args["--batches-per-step"] = 1
        py_args["--epochs"] = 1
        test_util.run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
