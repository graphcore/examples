# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent.joinpath("tut2_loops_data_pipeline")


class TestComplete(SubProcessChecker):

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_complete_ipu(self):
        self.run_command("python example_2.py",
                         working_path,
                         "Program ran successfully")
