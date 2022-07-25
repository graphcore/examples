# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import os
from pathlib import Path
from subprocess import run

import test_utils

custom_ops_loaded = False


@pytest.fixture
def custom_ops():
    if not custom_ops_loaded:
        test_utils.load_custom_sparse_logsoftmax_op()
        test_utils.load_custom_rnnt_op()
        return "ops loaded"
    else:
        return "ops loaded"
