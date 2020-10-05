# Copyright 2020 Graphcore Ltd.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        ''' Compile the tutorial code '''
        self.run_command("make clean", working_path, [])
        self.run_command("make all", working_path, [])

    def tearDown(self):
        self.run_command("make clean", working_path, [])

    @pytest.mark.category1
    def test_run_complete(self):
        ''' Check that the tutorial code runs '''

        self.run_command("./matrix-vector 10000 1000",
                         working_path,
                         ["Multiplying matrix of size 10000x1000 by vector of size 1000",
                          "Worst cost seen: 64373",
                          "Multiplication result OK"])
