# Copyright 2020 Graphcore Ltd.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent.joinpath("start_here")


class TestStartHere(SubProcessChecker):

    def setUp(self):
        ''' Compile the start here version of the tutorial code '''
        self.run_command("make clean", working_path, [])
        self.run_command("make all", working_path, [])

    def tearDown(self):
        self.run_command("make clean", working_path, [])

    @pytest.mark.category1
    def test_run_start_here(self):
        ''' Check that the start_here version of the tutorial code runs
            and exits with a non-zero return code'''

        with pytest.raises(AssertionError):
            self.run_command("./matrix-vector 40 50",
                             working_path, [])
