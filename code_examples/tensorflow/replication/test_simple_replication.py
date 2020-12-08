# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import unittest
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper


def run_simple_replication(**kwargs):
    """Helper function to run replication tensorflow python script with
       command line arguments"""
    out = run_python_script_helper(
        os.path.dirname(__file__), "simple_replication.py", **kwargs
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

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_replication_2(self):
        self._replication_factor_test_helper(2)

    @pytest.mark.ipus(4)
    @pytest.mark.category1
    def test_replication_4(self):
        self._replication_factor_test_helper(4)

    @pytest.mark.ipus(8)
    @pytest.mark.category1
    def test_replication_8(self):
        self._replication_factor_test_helper(8)

    @pytest.mark.ipus(4)
    @pytest.mark.category1
    def test_dataset_shape_less_than_replication_factor_3(self):
        self._replication_factor_test_helper(4, num_features=3)

    @pytest.mark.ipus(4)
    @pytest.mark.category1
    def test_dataset_shape_less_than_replication_factor_1(self):
        self._replication_factor_test_helper(4, num_features=1)

    @pytest.mark.ipus(4)
    @pytest.mark.category1
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

