# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import regex as re
from math import isclose

from examples_tests.test_util import SubProcessChecker
from tests.utils import get_app_root_dir


class TestBuildAndRun(SubProcessChecker):
    def test_run_cluster_gcn_sparse(self):
        cmd = "python run_cluster_gcn.py tests/train_small_graph_sparse.json --training.replicas 2"
        poprun_prefix = 'poprun --only-output-from-instance 0 --num-instances 2 --num-replicas 2 --ipus-per-replica 2'
        poprun_cmd = poprun_prefix + ' ' + cmd
        print(f"Running: {poprun_cmd}")
        self.run_command(poprun_cmd,
                         get_app_root_dir(),
                         [""])

    def test_run_cluster_gcn_dense(self):
        cmd = "python run_cluster_gcn.py tests/train_small_graph_dense.json --training.replicas 2"
        poprun_prefix = 'poprun --only-output-from-instance 0 --num-instances 2 --num-replicas 2 --ipus-per-replica 2'
        poprun_cmd = poprun_prefix + ' ' + cmd
        print(f"Running: {poprun_cmd}")
        self.run_command(poprun_cmd,
                         get_app_root_dir(),
                         [""])
