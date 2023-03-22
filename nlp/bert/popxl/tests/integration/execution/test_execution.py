# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path
from examples_tests.test_util import SubProcessChecker
import os
import sys
import pytest

root_dir = Path(__file__).parent.parent.parent.parent.resolve()


def bert_root_env_path():
    env = os.environ
    env["PYTHONPATH"] = ":".join((*sys.path, str(root_dir)))
    return env


class TestSquad(SubProcessChecker):
    def test_inference_phased(self):
        self.run_command(
            "python3 squad_inference.py  --layers 3 "
            "--device_iterations 10 --micro_batch_size 2 --data_parallel 1 "
            "--vocab_size 128 --sequence_length 8 --max_positional_length 8 "
            "--hidden_size 64 --heads 4",
            root_dir,
            ["Duration"],
            env=bert_root_env_path(),
        )

    def test_training_phased(self):
        self.run_command(
            "python3 squad_training.py  --layers 3 "
            "--micro_batch_size 2 --data_parallel 2 "
            "--vocab_size 128 --sequence_length 8 --max_positional_length 8 "
            "--hidden_size 64 --heads 4",
            root_dir,
            ["Duration"],
            env=bert_root_env_path(),
        )


class TestPretraining(SubProcessChecker):
    def test_training_phased(self):
        # TODO remove when https://phabricator.sourcevertex.net/T68087 is solved
        try:
            self.run_command(
                "python3 pretraining.py  --layers 3 "
                "--global_batch_size 16 --micro_batch_size 2 --data_parallel 2 "
                "--vocab_size 128 --sequence_length 8 --max_positional_length 8 "
                "--hidden_size 64 --heads 4",
                root_dir,
                ["Duration"],
                env=bert_root_env_path(),
            )
        except:
            self.run_command(
                "python3 pretraining.py  --layers 3 "
                "--global_batch_size 16 --micro_batch_size 2 --data_parallel 2 "
                "--vocab_size 128 --sequence_length 8 --max_positional_length 8 "
                "--hidden_size 64 --heads 4",
                root_dir,
                ["Duration"],
                env=bert_root_env_path(),
            )
