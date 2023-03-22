# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
import sys
from pathlib import Path

from examples_tests.test_util import SubProcessChecker

root_dir = Path(__file__).parent.parent.parent.parent.resolve()


def bloom_root_env_path():
    env = os.environ
    env["PYTHONPATH"] = ":".join((*sys.path, str(root_dir)))
    return env


class TestExecution(SubProcessChecker):
    def test_inference(self):
        self.run_command(
            "python3 inference.py --config bloom_560M_pod16 --layers 2 "
            "--tensor_parallel_1 2 --tensor_parallel_2 2 "
            "--vocab_size 128 --sequence_length 8 "
            "--hidden_size 64 --heads 4",
            root_dir,
            ["Duration"],
            env=bloom_root_env_path(),
        )
