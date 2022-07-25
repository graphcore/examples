# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestIpuFunction(SubProcessChecker):

    def _run_test_ipu_func(self, cmd_args, extra_env={}):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        env.update(extra_env)
        self.run_command(cmd_args, build_dir, ["All asserts pass."], env=env)

    @pytest.mark.ipus(1)
    def test_sparse_ipu_function(self):
        # Test default parameters:
        self._run_test_ipu_func("python3 ipu_sparse_ops/tools/sparse_ipu_function.py")
