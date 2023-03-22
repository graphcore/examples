# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from tutorials_tests import testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category2
@pytest.mark.ipus(1)
def test_run_default_ipu():
    # Check default params
    testing_util.run_command("python anchor_tensor_example.py", working_path, "Saved histogram")
