# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path

import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
import tutorials_tests.testing_util as testing_util

working_path = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def setUp():
    testing_util.run_command("make", working_path, [])


@pytest.mark.ipus(1)
@pytest.mark.category1
def test_run_prefetch():
    testing_util.run_command("./prefetch", working_path, ["Running", "complete", "prefetch"])
