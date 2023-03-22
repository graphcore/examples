# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from tests.utils import get_app_root_dir
from examples_tests.test_util import SubProcessChecker


class TestBuildAndRun(SubProcessChecker):
    def _get_glue_command(self, extra_args=None):
        base_cmd = "python run_seq_classification.py --config tests/glue_tiny_test.json"
        print(f"Running: {base_cmd}")
        if extra_args is not None and len(extra_args) > 0:
            base_cmd += " " + " ".join(extra_args)
        return base_cmd

    def test_run_GLUE(self):
        cmd = self._get_glue_command(extra_args=["--generated-dataset", "true"])
        self.run_command(cmd, get_app_root_dir(), [""])
