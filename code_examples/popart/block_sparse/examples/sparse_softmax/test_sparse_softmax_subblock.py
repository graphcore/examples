# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import unittest
import re

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper, run_test_helper


def run_sparse_softmax_subblock(**kwargs):
    """Helper function to run popart sparse softmax with subblock model python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__), "sparse_softmax_subblock_demo.py",
                                   **kwargs)
    return out


class Test(unittest.TestCase):


    @classmethod
    @pytest.mark.ipus(1)
    @pytest.mark.category2
    def test_output(self):
        """Generic test on default arguments in training"""
        out = run_test_helper(
            run_sparse_softmax_subblock
        )
