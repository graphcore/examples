# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestBuildAndRun(SubProcessChecker):
    def _run_test_update(self, cmd_args):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        self.run_command(cmd_args, build_dir, ["All results match"], env=env)

    @pytest.mark.ipus(1)
    def test_fc_sparsity_update(self):
        # Test default parameters:
        self._run_test_update("python3 ipu_sparse_ops/tools/fc_update.py ")
