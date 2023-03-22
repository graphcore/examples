# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category2
def test_run_demo():
    testing_util.run_command("python3 demo.py", working_path, "128/128")
