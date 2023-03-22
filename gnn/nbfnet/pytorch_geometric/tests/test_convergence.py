# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pathlib
import regex as re

from examples_tests.test_util import SubProcessChecker


EXAMPLE_ROOT_DIR = pathlib.Path(__file__).parents[1].resolve()


class TestBuildAndRun(SubProcessChecker):
    def test_model(self):
        cmd = "python run_nbfnet.py --c configs/IndFB15k-237_v1.yaml"
        print(f"Running: {cmd}")
        results = self.run_command(cmd, EXAMPLE_ROOT_DIR, [""])
        filtered_results = re.findall(
            r"(?:test_MRR: )(\d.\d*)|"
            r"(?:test_hits@1: )(\d.\d*)|"
            r"(?:test_hits@3: )(\d.\d*)|"
            r"(?:test_hits@10: )(\d.\d*)",
            results,
        )
        EXPECTED_SCORES = [0.35, 0.25, 0.35, 0.45]
        for i, (expect, found) in enumerate(zip(EXPECTED_SCORES, filtered_results)):
            assert float(expect) < float(found[i])
