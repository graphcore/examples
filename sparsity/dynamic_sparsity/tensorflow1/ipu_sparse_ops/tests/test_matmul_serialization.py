# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class Testmatmul_serialization(SubProcessChecker):

    def _run_test_matmul(self, cmd_args, extra_env={}):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        env.update(extra_env)
        self.run_command(cmd_args, build_dir, ["All results match."], env=env)

    def test_matmul_serialization_dim_0(self):
        # Test default parameters:
        self._run_test_matmul("./ipu_sparse_ops/tests/test_matmul_serialization 32 16 8 2 1")

    def test_matmul_serialization_dim_1(self):
        # Test default parameters:
        self._run_test_matmul("./ipu_sparse_ops/tests/test_matmul_serialization 32 16 8 1 2")

    def test_matmul_serialization_2D(self):
        # Test default parameters:
        self._run_test_matmul("./ipu_sparse_ops/tests/test_matmul_serialization 32 16 8 2 2")
