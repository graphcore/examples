# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import sys
import pytest

run_dir = Path(__file__).parent.parent


@pytest.mark.usefixtures("custom_ops")
class TestBuildAndRun(SubProcessChecker):

    def _run_command(self, args=""):
        cmd = sys.executable + " tests/sharded_embedding_tool.py "
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        self.run_command(f"{cmd} {args}",
                         run_dir,
                         ["Results match."], env=env)

    @pytest.mark.ipus(2)
    def test_small_embedding(self):
        self._run_command("--ipus 2")

    @pytest.mark.ipus(4)
    def test_larger_embedding(self):
        self._run_command("--ipus 4 --vocab-size 8000 --feature-size 768 --sequence-length 256")
