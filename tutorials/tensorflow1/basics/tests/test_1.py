# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent.joinpath("tut1_porting_a_model")


class TestComplete(SubProcessChecker):

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_complete_ipu(self):
        self.run_command("python example_1.py",
                         working_path,
                         "Program ran successfully")
