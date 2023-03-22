# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from tests.utils import get_app_root_dir
from examples_tests.test_util import SubProcessChecker


class TestBuildAndRun(SubProcessChecker):
    def _get_squad_command(self, extra_args=None):
        base_cmd = "python run_squad.py --config tests/squad_tiny_test.json"
        print(f"Running: {base_cmd}")
        if extra_args is not None and len(extra_args) > 0:
            base_cmd += " " + " ".join(extra_args)
        return base_cmd

    def test_run_squad(self):
        cmd = self._get_squad_command(extra_args=["--generated-dataset", "true"])
        self.run_command(cmd, get_app_root_dir(), ["Evaluation metrics:"])
