#!/usr/bin/python
# Copyright 2019 Graphcore Ltd.

import inspect
import os
import pytest
import unittest

import tests.test_util as test_util


def run_simple_replication(**kwargs):
    """Helper function to run replication tensorflow python script with
       command line arguments"""
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    out = test_util.run_python_script_helper(
        cwd, "simple_replication.py", **kwargs
    )
    return out


class TestSimpleReplicationModelling(unittest.TestCase):
    """High-level integration tests for simple replication Tensorflow
       example"""

    @classmethod
    def setUpClass(cls):
        cls.generic_arguments = {
            "--replication-factor": 2,
            "--num-data-points": 50,
            "--num-features": 100,
            "--num-iters": 250
        }

    def test_replication_2(self):
        self._replication_factor_test_helper(2)

    def test_replication_4(self):
        self._replication_factor_test_helper(4)

    def test_replication_8(self):
        self._replication_factor_test_helper(8)

    def test_dataset_shape_less_than_replication_factor(self):
        self._replication_factor_test_helper(4, num_features=3)

    def test_dataset_shape_less_than_replication_factor(self):
        self._replication_factor_test_helper(4, num_features=1)

    def test_dataset_shape_equal_to_replication_factor(self):
        self._replication_factor_test_helper(4, num_features=4)

    def _replication_factor_test_helper(
        self, factor, num_features=None
    ):
        """Helper function for running tests of varying replication
           factor"""

        py_args = self.generic_arguments.copy()
        py_args["--replication-factor"] = str(factor)

        if num_features:
            py_args["--num-features"] = num_features

        run_simple_replication(**py_args)


if __name__ == "__main__":
    unittest.main()
