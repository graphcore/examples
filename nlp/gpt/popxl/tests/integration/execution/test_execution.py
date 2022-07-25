# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path
from examples_tests.test_util import SubProcessChecker
import os
import sys

root_dir = Path(__file__).parent.parent.parent.parent.resolve()


def gpt_root_env_path():
    env = os.environ
    env["PYTHONPATH"] = ":".join((*sys.path, str(root_dir)))
    return env


class TestPretraining(SubProcessChecker):
    def test_training_phased(self):
        self.run_command(
            "python3 execution/pretraining.py  --layers 3 "
            "--global_batch_size 16 --micro_batch_size 2 --data_parallel 2 --tensor_parallel 2 "
            "--vocab_size 128 --sequence_length 8 --max_positional_length 8 "
            "--hidden_size 64 --heads 4",
            root_dir,
            ["Duration"],
            env = gpt_root_env_path())
