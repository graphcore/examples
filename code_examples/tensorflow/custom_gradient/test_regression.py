# Copyright 2020 Graphcore Ltd.
from tests.test_util import SubProcessChecker
from pathlib import Path

build_dir = Path(__file__).parent


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        self.run_command("make", build_dir, [])

    def tearDown(self):
        self.run_command("make clean", build_dir, [])

    def test_run_regression(self):
        self.run_command("python3 regression.py",
                         build_dir,
                         ["Losses, grads and weights match."])
