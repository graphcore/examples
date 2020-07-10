# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from tests.test_util import SubProcessChecker
from pathlib import Path
import os

build_dir = Path(__file__).parent.parent.parent


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        self.run_command("make -j", build_dir, [])

    def tearDown(self):
        self.run_command("make clean", build_dir, [])

    def test_sparse_matmul_tools(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        # Test default parameters:
        self.run_command("python3 ipu_sparse_ops/tools/sparse_matmul.py",
                         build_dir,
                         ["Results match."], env=env)
        # Test the random pattern option:
        self.run_command("python ipu_sparse_ops/tools/sparse_matmul.py --batch-size 16 --input-size 64 --output-size 32 --pattern random --density 0.05",
                         build_dir,
                         ["Results match."], env=env)
        # Test an MNIST FC sized matmul:
        self.run_command("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 784 --output-size 300 --batch-size 16",
                         build_dir,
                         ["Results match."], env=env)
        # Test a larger size:
        self.run_command("python3 ipu_sparse_ops/tools/sparse_matmul.py --input-size 8192 --output-size 1536 --batch-size 8",
                         build_dir,
                         ["Results match."], env=env)
