# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

build_dir = Path(__file__).parent.parent


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        self.run_command("make", build_dir, [])

    @pytest.mark.category1
    def test_run_ipu_model(self):
        self.run_command("./example --model",
                         build_dir,
                         ["Results match.", "Using IPU model"])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_run_ipu(self):
        self.run_command("./example",
                         build_dir,
                         ["Using HW device ID", "Results match."])
