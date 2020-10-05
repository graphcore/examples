# Copyright 2020 Graphcore Ltd.
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
        ''' Check that the start here version of the tutorial code runs '''

        self.run_command("./tut2_start_here",
                         working_path,
                         ["Program complete"])

    @pytest.mark.category1
    def test_run_complete(self):
        ''' Check that the complete version of the tutorial code runs '''

        self.run_command("./tut2_complete",
                         working_path,
                         ["Program complete",
                          r"v3: {\W+{5,4.5},\W+{4,3.5}",
                          r"v4: {\W+{9,7.5},\W+{6,4.5}",
                          r"v5: {\W+{5,3.5},\W+{5,3.5}"])
