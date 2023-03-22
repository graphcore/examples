# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from pathlib import Path
from examples_tests.test_util import SubProcessChecker
import os
import sys

root_dir = Path(__file__).parent.parent.parent.parent.resolve()


def gptj_root_env_path():
    env = os.environ
    env["PYTHONPATH"] = ":".join((*sys.path, str(root_dir)))
    return env


class TestPretraining(SubProcessChecker):
    def test_finetuning(self):
        self.run_command(
            "python3 finetuning.py --config tiny --layers 3 "
            "--global_batch_size 16 --micro_batch_size 2 --data_parallel 2 --tensor_parallel 2 "
            "--vocab_size 128 --sequence_length 8 --rotary_dim 16 "
            "--hidden_size 64 --heads 4",
            root_dir,
            ["Duration"],
            env=gptj_root_env_path(),
        )

    def test_inference(self):
        self.run_command(
            "python3 inference.py --config tiny --layers 3 "
            "--micro_batch_size 16 --data_parallel 1 --tensor_parallel 2 "
            "--vocab_size 128 --sequence_length 8 --rotary_dim 16 "
            "--hidden_size 64 --heads 4",
            root_dir,
            ["Duration"],
            env=gptj_root_env_path(),
        )
