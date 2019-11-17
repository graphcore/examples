# Copyright 2019 Graphcore Ltd.
import inspect
import unittest
import os
import sys
import subprocess
from contextlib import contextmanager

import tests.test_util as tu


def run_multi_ipu(shards, batch_size, batches_per_step):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    py_version = "python" + str(sys.version_info[0])
    cmd = [py_version, "multi_ipu.py",
           "--shards", str(shards),
           "--batch-size", str(batch_size),
           "--batches-per-step", str(batches_per_step)]
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
    print(out)
    return out


class TestMultiIPUPopART(unittest.TestCase):
    """Tests for multi-IPU popART code example"""

    @classmethod
    def setUpClass(cls):
        pass

    # Multi-IPU tests

    def test_multi_ipu_2_10(self):
        out = run_multi_ipu(shards=2, batch_size=10, batches_per_step=100000)


if __name__ == '__main__':
    unittest.main()
