# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""Tests for the AdGMoM reimplementation."""
import pytest
from examples_tests.test_util import SubProcessChecker
from pathlib import Path


current_path = Path(__file__).parent


class AdgmomTest(SubProcessChecker):
    """Test simple command line executions."""
    @pytest.mark.ipus(2)
    def test_default(self):
        """Test the default"""
        self.run_command(
            "python tf2_AdGMoM.py", current_path, ["Iterations", "6000"])
