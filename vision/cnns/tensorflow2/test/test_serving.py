# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
from examples_tests.test_util import SubProcessChecker
from test_common import run_serving


class Simple(SubProcessChecker):
    def test_simple_serving_bs_1(self):
        run_serving(
            self,
            "--config",
            "resnet50_infer_test",
            "--synthetic-data",
            "host",
            "--batch-size",
            "1",
            "--port",
            "8502",
            "--num-threads",
            "1",
            "--pytest",
            "--verbose",
        )
