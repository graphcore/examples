# Copyright 2020 Graphcore Ltd.
import os
import pytest
import re
import subprocess
import unittest

import numpy as np

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper


def run_simple_replication(**kwargs):
    """Helper function to run TensorFlow replication python script with
       command line arguments"""
    out = run_python_script_helper(
        os.path.dirname(__file__), "replication.py", **kwargs
    )
    return out


class TestReplicationTensorFlow(unittest.TestCase):
    """High-level integration tests for TensorFlow replication example"""

    @classmethod
    def setUpClass(cls):
        cls.generic_arguments = {
            "--replicas": 2,
            "--max-cross-replica-sum-buffer-size": 10*1024**2,
            "--batch-size": 32,
            "--iterations-per-step": 250,
            "--batches-to-accumulate": 1,
            "--learning-rate": 0.001,
            "--steps": 1,
        }

    def _replication_factor_test_helper(self, **kwargs):
        """Helper function for calling replication script with varying cli
           arguments"""
        py_args = self.generic_arguments.copy()
        for kwarg, value in kwargs.items():
            cli_name = '--' + kwarg.replace('_', '-')
            py_args[cli_name] = str(value)
        return run_simple_replication(**py_args)

    def _check_convergence_helper(self, out):
        """
        Helper function to check if the model is converging with its output
        Assumes a specific output format:
            Step: X | Average loss: Y
        """
        assert "Average loss" in out, ("String 'Average loss' not found in"
                                       " script output. Did the format of the"
                                       " script output change?")
        # Parse the losses from the program output with regex
        losses = list(map(float, re.findall(r'Average loss: (\d+\.?\d*)', out)))
        # Get the sign of the loss's derivative
        diff = [np.sign(y - x) for x, y in zip(losses[:-1], losses[1:])]
        # The derivative should be mostly (majority) negative
        assert sum(diff) < 0, "Model not converging smoothly/at all."

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_no_replicas_convergence(self):
        """
        Test model converges with no replicas
        Expected: Average loss slowly decreases over time more than it increases
        """
        out = self._replication_factor_test_helper(replicas=1, steps=40)
        self._check_convergence_helper(out)

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_2_replicas_convergence(self):
        """
        Test model converges with 2 replicas
        Expected: Average loss slowly decreases over time more than it increases
        """
        out = self._replication_factor_test_helper(steps=40)
        self._check_convergence_helper(out)

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_replication_synthetic(self):
        """ Test model compiles when using synthetic data """
        # 'run_python_script_helper' needs flag arguments to be passed with
        # empty strings
        self._replication_factor_test_helper(use_synthetic_data='')

    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_no_replication(self):
        """ Test model compiles when using no replication """
        self._replication_factor_test_helper(replicas=1)

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_2_replicas(self):
        """ Test model compiles when using 2 replicas """
        self._replication_factor_test_helper(replicas=2)

    @pytest.mark.ipus(4)
    @pytest.mark.category2
    def test_4_replicas(self):
        """ Test model compiles when using 4 replicas """
        self._replication_factor_test_helper(replicas=4)

    @pytest.mark.ipus(8)
    @pytest.mark.category2
    def test_8_replicas(self):
        """ Test model compiles when using 8 replicas """
        self._replication_factor_test_helper(replicas=8)

    @pytest.mark.ipus(16)
    @pytest.mark.category2
    def test_16_replicas(self):
        """ Test model compiles when using 16 replicas """
        self._replication_factor_test_helper(replicas=16)

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_replication_with_0_buffer(self):
        """
        Test model compiles when using a max cross replica sum buffer
        size of 0
        """
        self._replication_factor_test_helper(
            max_cross_replica_sum_buffer_size=0
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_replication_with_BS_1(self):
        """ Test model compiles with batch size 1 """
        self._replication_factor_test_helper(batch_size=1)

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_replication_with_8_gradient_accumulation(self):
        """ Test model compiles with gradient accumulation on """
        self._replication_factor_test_helper(
            batches_to_accumulate=8,
            iterations_per_step=256
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_replication_with_fewer_iterations_than_accumulation(self):
        """
        Test model correctly fails when requesting to accumulate more batches
        than there are iterations
        """
        with pytest.raises(subprocess.CalledProcessError) as e:
            self._replication_factor_test_helper(
                batches_to_accumulate=8,
                iterations_per_step=4
            )

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_replication_with_indivisible_iterations_and_accumulation(self):
        """
        Test model correctly fails when requesting to accumulate a number of
        batches not divisible by the number of iterations
        """
        with pytest.raises(subprocess.CalledProcessError):
            out = self._replication_factor_test_helper(
                batches_to_accumulate=8,
                iterations_per_step=251
            )
