#!/usr/bin/python
# Copyright 2019 Graphcore Ltd.

import inspect
import os
import sys
import pytest
import unittest

import utils.tests.test_util as test_util


def run_ipu_estimator_cnn(**kwargs):
    """Helper function to run ipu_estimator_cnn tensorflow python script with
       command line arguments"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    out = test_util.run_python_script_helper(
        cwd, "ipu_estimator_cnn.py", **kwargs
    )
    return out


class TestIPUEstimatorCNN(unittest.TestCase):
    """High-level integration tests for ipu_estimator_cnn Tensorflow
       example"""

    @classmethod
    def setUpClass(cls):
        cls.generic_arguments = {
            "--batch-size": 32,
            "--batches-per-step": 100,
            "--epochs": 1,
            "--learning-rate": 0.01,
            "--log-interval": 10,
            "--summary-interval": 1,
        }

    def test_normal_usage(self):
        self._ipu_estimator_cnn_test_helper()

    def test_strange_numbers(self):
        self._ipu_estimator_cnn_test_helper(batch_size=13, batches_per_step=89)

    def test_test_only(self):
        self._ipu_estimator_cnn_test_helper(model_dir='/tmp/tmpje829e90/')
        self._ipu_estimator_cnn_test_helper(model_dir='/tmp/tmpje829e90/', test_only='')

    def test_profile(self):
        self._ipu_estimator_cnn_test_helper(model_dir='/tmp/tmpid930d30/', profile='')
        assert os.path.exists('/tmp/tmpid930d30/train_report.txt'), "Train report wasn't generated"
        assert os.path.exists('/tmp/tmpid930d30/eval_report.txt'), "Eval report wasn't generated"

    def _ipu_estimator_cnn_test_helper(self, **kwargs):
        """Helper function for running tests of varying arguments"""

        py_args = self.generic_arguments.copy()

        for kwarg in kwargs.keys():
            kwargadj = '--' + kwarg.replace('_', '-')
            py_args[kwargadj] = kwargs[kwarg]

        run_ipu_estimator_cnn(**py_args)


if __name__ == "__main__":
    unittest.main()
