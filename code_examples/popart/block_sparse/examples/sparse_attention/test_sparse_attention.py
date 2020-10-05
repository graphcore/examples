
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import unittest
import re

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, run_test_helper


def run_sparse_attention(**kwargs):
    """Helper function to run popart sparse attention model python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__), "short_demo.py",
                                   **kwargs)
    return out


class TestSparseAttention(unittest.TestCase):


    @classmethod
    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_output(self):
        """Generic test on default arguments in training"""
        out = run_test_helper(
            run_sparse_attention
        )

        prob = None
        grad = None

        prob_expected = 0.00098
        prob_tol = 0.00001
        grad_expected = 2e-11
        grad_tol = 1e-11

        for line in out.split("\n"):
            if re.match(r"Probability mean (-?[\d]+(?:.[\d]+)?(?:e[+-]?\d+)?)", line):
                prob = float(re.match(r"Probability mean (-?[\d]+(?:.[\d]+)?(?:e[+-]?\d+)?)", line).groups()[0])
            elif re.match(r"Logits grad mean (-?[\d]+(?:.[\d]+)?(?:e[+-]?\d+)?)", line):
                grad = float(re.match(r"Logits grad mean (-?[\d]+(?:.[\d]+)?(?:e[+-]?\d+)?)", line).groups()[0])
            if prob is None:
                assert False, "'Probability mean {}' pattent was not found in  the output"
            else:
                assert abs(prob_expected - prob) < prob_tol
            if grad is None:
                assert False, "'Logits grad mean {}' pattent was not found in  the output"
            else:
                assert abs(grad_expected - grad) < grad_tol
