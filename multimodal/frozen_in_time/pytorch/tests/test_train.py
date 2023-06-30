# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Tests for the Frozen implementation."""
import sys
from pathlib import Path
import pytest
from examples_tests.test_util import SubProcessChecker

frozen_root_path = str(Path(__file__).absolute().parent.parent)
sys.path.append(frozen_root_path)


class FrozenTest(SubProcessChecker):

    """Test training command line executions."""

    @pytest.mark.category3
    @pytest.mark.ipus(2)
    def test_train(self):
        """Test the training on 2 IPUs"""
        self.run_command("python3 run.py --config_name configs/unit_test.json", frozen_root_path, ["Training complete"])
