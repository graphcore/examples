# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import regex as re
from math import isclose

from examples_tests.test_util import SubProcessChecker
from tests.utils import get_app_root_dir


class TestBuildAndRun(SubProcessChecker):
    def test_run_cluster_gcn_sparse(self):
        cmd = "python run_cluster_gcn.py tests/train_small_graph_sparse.json"
        print(f"Running: {cmd}")
        self.run_command(cmd, get_app_root_dir(), [""])

    def test_run_cluster_gcn_dense(self):
        cmd = "python run_cluster_gcn.py tests/train_small_graph_dense.json"
        print(f"Running: {cmd}")
        self.run_command(cmd, get_app_root_dir(), [""])

    @pytest.mark.long_test
    def test_run_cluster_gcn_arxiv(self):
        cmd = "python run_cluster_gcn.py configs/train_arxiv.json --wandb false --data-path ."
        print(f"Running: {cmd}")
        results = self.run_command(cmd, get_app_root_dir(), [""])
        filtered_results = re.findall(
            r"(?:Validation Accuracy: )(\d.\d*)|"
            r"(?:Validation F1.macro: )(\d.\d*)|"
            r"(?:Validation F1.micro: )(\d.\d*)|"
            r"(?:Test Accuracy: )(\d.\d*)|"
            r"(?:Test F1.macro: )(\d.\d*)|"
            r"(?:Test F1.micro: )(\d.\d*)",
            results,
        )
        EXPECTED_SCORES = [0.55, 0.35, 0.55, 0.55, 0.35, 0.55]
        for i, (expect, found) in enumerate(zip(EXPECTED_SCORES, filtered_results)):
            assert float(expect) < float(found[i])
