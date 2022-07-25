# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent


class TestET0(SubProcessChecker):

    def setUp(self):
        self.run_command("sh get_data.sh",
                         working_path,
                         ["Unpacking dummy data of et0 and make the custom operaton"])

    @pytest.mark.ipus(1)
    def test_et0(self):
        self.run_command("python3 test_chinese_map/kriging_test.py",
                         working_path,
                         ["kriging start",
                          "Complete the calculation of et0 in 1 day"])
