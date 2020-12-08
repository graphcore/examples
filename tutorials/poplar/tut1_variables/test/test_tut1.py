# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        ''' Compile the start here and complete versions of the tutorial code '''
        self.run_command("make clean", working_path, [])
        self.run_command("make all", working_path, [])

    def tearDown(self):
        self.run_command("make clean", working_path, [])

    @pytest.mark.category1
    def test_run_start_here(self):
        ''' Check that the start here version of the tutorial code
            for the IPU Model runs '''

        self.run_command("./tut1_start_here",
                         working_path,
                         [])

    @pytest.mark.category1
    def test_run_ipu_model(self):
        ''' Check that the complete version of the tutorial code
            for the IPU Model runs '''

        self.run_command("./tut1_ipu_model_complete",
                         working_path,
                         ["Program complete", "h3 data:", "0 1 1.5 2",
                          "v4-1: {10,11,12,13,14,15,16,17,18,19}"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_ipu_hardware(self):
        ''' Check that the complete version of the tutorial code
            for the IPU hardware runs '''
        self.run_command("./tut1_ipu_hardware_complete",
                         working_path,
                         ["Attached to IPU", "Program complete",
                          "h3 data:", "0 1 1.5 2",
                          "v4-1: {10,11,12,13,14,15,16,17,18,19}"])
