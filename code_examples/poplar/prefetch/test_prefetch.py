# Copyright 2020 Graphcore Ltd.
import os

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        self.run_command("make", working_path, [])

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_run_prefetch(self):
        self.run_command("./prefetch",
                         working_path,
                         ["Running", "complete", "prefetch"])
