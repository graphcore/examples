# Copyright 2020 Graphcore Ltd.
import os
import subprocess
import sys
import unittest
import pytest

import numpy as np
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import assert_result_equals_tensor_value


def run_simple_sharding(autoshard):
    py_version = "python{}".format(sys.version_info[0])
    cmd = [py_version, "simple_sharding.py"]
    if autoshard:
        cmd.append("--autoshard")
    out = subprocess.check_output(cmd, cwd=os.path.dirname(__file__),
                                  universal_newlines=True)
    return out


class TestTensorFlowSharding(unittest.TestCase):
    """High-level integration tests for tensorflow sharding examples"""

    @classmethod
    def setUpClass(cls):
        pass

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_manual_sharding(self):
        """Manual sharding example using 2 shards"""
        out = run_simple_sharding(False)
        assert_result_equals_tensor_value(
            out, np.array([3.0, 8.0], dtype=np.float32)
        )

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_auto_sharding(self):
        """Automatic sharding example using 2 shards"""
        out = run_simple_sharding(True)
        assert_result_equals_tensor_value(
            out, np.array([3.0, 8.0], dtype=np.float32)
        )
