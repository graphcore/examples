# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from examples_tests.test_util import SubProcessChecker
from tests.utils import get_app_root_dir


class TestBuildAndRun(SubProcessChecker):
    def test_run_cluster_gcn(self):
        cmd = "python run_cluster_gcn.py tests/train_small_graph.json"
        print(f"Running: {cmd}")
        self.run_command(cmd,
                         get_app_root_dir(),
                         [""])
