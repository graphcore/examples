# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestBuildAndRun(SubProcessChecker):

    def _run_test_embed(self, cmd_args, extra_env={}):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        env.update(extra_env)
        self.run_command(cmd_args, build_dir, ["Results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_sparse_embedding_default_projection_only(self):
        # Test default parameters:
        self._run_test_embed("python3 ipu_sparse_ops/tools/sparse_embedding.py --check-projection-grads-only")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_sparse_embedding_default(self):
        # Test default parameters:
        self._run_test_embed("python3 ipu_sparse_ops/tools/sparse_embedding.py")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_sparse_embedding_default_fp16(self):
        # Test default parameters:
        self._run_test_embed("python3 ipu_sparse_ops/tools/sparse_embedding.py --data-type fp16")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_sparse_gpt2_xl_s128_fp16(self):
        # Test default parameters:
        self._run_test_embed("python3 ipu_sparse_ops/tools/sparse_embedding.py --hidden-size 1600 --sequence-size 128 --data-type fp16 --pooling-type=SUM --embedding-size 51520")
