# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import os
import unittest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, run_test_helper, \
    parse_results_for_accuracy


def run_mnist_training(**kwargs):
    """Helper function to run popart block-sparse mnist linear model python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__), "bs_mnist.py",
                                   **kwargs)
    return out


class TestMNIST(unittest.TestCase):
    """High-level integration tests for training with the MNIST data-set"""

    @classmethod
    def setUpClass(cls):
        cls.accuracy_tolerances = 3.0
        cls.generic_arguments = {
            "--batches-per-step": 4,
            "--hidden-size": 16,
            "--sparsity-level": 0.7,
            "--fix-seed": "",
            "./data": ""
        }


    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_mnist_train(self):
        """Generic test on default arguments in training"""
        py_args = self.generic_arguments.copy()
        out = run_test_helper(
            run_mnist_training,
            **py_args
        )
        expected_accuracy = [
            50.32, 58.67, 68.09, 72.81, 75.30, 77.14, 78.53, 79.75, 80.62, 81.37
        ]
        parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )


    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_mnist_train_sharded_pipelined(self):
        """Generic test on default arguments in training"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 2
        py_args["--pipeline"] = ""
        out = run_test_helper(
            run_mnist_training,
            **py_args
        )
        expected_accuracy = [
            50.32, 58.67, 68.09, 72.81, 75.30, 77.14, 78.53, 79.75, 80.62, 81.37
        ]
        parse_results_for_accuracy(
            out, expected_accuracy, self.accuracy_tolerances
        )
