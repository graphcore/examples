# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import os
import unittest

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, run_test_helper, \
    parse_results_for_accuracy


def run_popart_mnist_training(**kwargs):
    """Helper function to run popart mnist linear model python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__), "popart_mnist.py",
                                   **kwargs)
    return out


class TestPopARTMNISTImageClassification(unittest.TestCase):
    """High-level integration tests for training with the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cls.accuracy_tolerances = 3.0
        cls.generic_arguments = {
            "--batch-size": 4,
            "--batches-per-step": 1000,
            "--epochs": 10,
            "--num-ipus": 1,
            "--validation-final-epoch": "",
        }

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_train(self):
        """Generic test on default arguments in training"""
        py_args = self.generic_arguments.copy()
        out = run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [90.70]
        parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_sharded(self):
        """Generic test on default arguments in training over 2 IPUs"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 2
        out = run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [90.70]
        parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_replicated(self):
        """Generic test on default arguments in training over 2 IPUs
           with replication"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 2
        py_args["--replication-factor"] = 2
        out = run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [90.70]
        parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_sharded_pipelined(self):
        """Generic test on default arguments in training over 2 IPUs
           and pipelined"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 2
        py_args["--pipeline"] = ""
        out = run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [89.25]
        parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )

    @pytest.mark.ipus(4)
    @pytest.mark.category2
    def test_mnist_train_replicated_pipelined(self):
        """Generic test on default arguments in training over 2 IPUs
           and pipelined"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 4
        py_args["--replication-factor"] = 2
        py_args["--pipeline"] = ""
        out = run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
        expected_accuracy = [89.25]
        parse_results_for_accuracy(
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
        run_test_helper(
            run_popart_mnist_training,
            **py_args
        )

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_simulation(self):
        """Simulation test with basic arguments"""
        py_args = self.generic_arguments.copy()
        py_args["--simulation"] = ""
        py_args["--epochs"] = 1
        run_test_helper(
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
        run_test_helper(
            run_popart_mnist_training,
            **py_args
        )
