# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).parent.parent


@pytest.mark.category1
@pytest.mark.ipus(1)
def test_tf2_recompilation():
    testing_util.run_command("python3 TF2_recompilation.py", working_path, "Caching/warm up test")
