# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
import sys
from pathlib import Path

from examples_tests.test_util import SubProcessChecker

root_dir = Path(__file__).parent.parent.parent.parent.resolve()


def dolly_root_env_path():
    env = os.environ
    env["PYTHONPATH"] = ":".join((*sys.path, str(root_dir)))
    return env


class TestExecution(SubProcessChecker):
    def test_inference(self):
        self.run_command(
            "python3 inference.py --config tiny --layers 2 "
            "--tensor_parallel 4 "
            "--vocab_size 128 --sequence_length 16 "
            "--hidden_size 128 --heads 8",
            root_dir,
            ["Duration"],
            env=dolly_root_env_path(),
        )
