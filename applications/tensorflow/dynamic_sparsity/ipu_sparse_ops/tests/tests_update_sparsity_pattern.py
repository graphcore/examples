# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


class TestBuildAndRun(SubProcessChecker):
    def _run_test_update(self, cmd_args):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        self.run_command(cmd_args, build_dir, ["All results match"], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_fc_sparsity(self):
        # Test default parameters:
        self._run_test_update("python ipu_sparse_ops/tools/fc_update_test.py ")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_gru_sparsity(self):
        # Test default parameters:
        self._run_test_update("python ipu_sparse_ops/tools/gru_update_test.py ")
