# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import subprocess
import unittest

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import run_python_script_helper


def run_phased_execution(**kwargs):
    """Helper function to run phased execution python script with
       command line arguments"""
    out = run_python_script_helper(os.path.dirname(__file__),
                                   "phased_execution.py", **kwargs)
    return out


class TestPhasedExecutionPopART(unittest.TestCase):
    """Tests for phased execution popART code example"""

    @classmethod
    def setUpClass(cls):
        pass

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    @pytest.mark.requires_remote_buffers
    def test_phased_execution(self):
        """Test that the code runs with default arguments"""
        run_phased_execution()

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_sharded_execution(self):
        """Test that the code runs in sharded mode
            (i.e. no phased execution)"""
        py_args = {"--sharded-execution": ""}
        run_phased_execution(**py_args)
