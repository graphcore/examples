#!/usr/bin/python
# Copyright 2020 Graphcore Ltd.
import os
import unittest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import run_python_script_helper, run_test_helper, \
    parse_results_for_accuracy


def run_pytorch_Torchvision(**kwargs):
    return run_python_script_helper(os.path.dirname(__file__),
                                    "torchvision_examples.py",
                                    **kwargs)


class TestPytorchTorchvisionTraining(unittest.TestCase):
    """High-level integration tests for the Torchvision models training on
       the CIFAR-10 data-set"""

    @classmethod
    def setUpClass(cls):
        cls.time_tolerances = 0.2
        cls.generic_arguments = {
            "--no-progress-bar": "",
            "--no-shuffle": "",
            "--no-validation": ""
        }

    def test_train_torchvision(self):
        """Generic functional test"""
        py_args = self.generic_arguments.copy()
        py_args["--num-ipus"] = 2
        py_args["--epochs"] = 10
        out = run_test_helper(run_pytorch_Torchvision, **py_args)
        expected_accuracy = [
            16.85, 24.52, 30.77, 36.34, 40.65, 45.59, 49.07, 52.13, 55.52,
            58.14
        ]
        parse_results_for_accuracy(out, expected_accuracy, 2.5)
