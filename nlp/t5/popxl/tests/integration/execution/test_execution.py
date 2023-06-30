# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from pathlib import Path
from examples_tests.test_util import SubProcessChecker
import os
import sys

root_dir = Path(__file__).parent.parent.parent.parent.resolve()


def t5_root_env_path():
    env = os.environ
    env["PYTHONPATH"] = ":".join((*sys.path, str(root_dir)))
    return env


class TestExecution(SubProcessChecker):
    def test_finetuning(self):
        self.run_command(
            "python3 finetuning.py --config tiny --micro_batch_size 2",
            root_dir,
            ["Duration"],
            env=t5_root_env_path(),
        )

    def test_inference(self):
        self.run_command(
            "python3 inference.py --config tiny --micro_batch_size 8",
            root_dir,
            ["Duration"],
            env=t5_root_env_path(),
        )
