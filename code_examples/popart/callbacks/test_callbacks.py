# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import unittest
import callbacks
import pytest
import os
from examples_tests.test_util import SubProcessChecker


class TestCallbacks(SubProcessChecker):
    """Tests for the popART LSTM synthetic benchmarks"""

    @pytest.mark.category1
    def test_example_runs(self):
        working_path = os.path.dirname(__file__)
        self.run_command(
            "python3 callbacks.py --data-size 1000", working_path, ["Mul:0", "Add:0"]
        )
