# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestReadmeCommands(SubProcessChecker):

    @pytest.mark.ipus(8)
    def test_reinforcement_learning_model(self):
        self.run_command("python3 rl_benchmark.py --micro_batch_size 8 --time_steps 16 --num_ipus 8",
                         working_path,
                         [r"Average (\w+.\w+) items/sec"])
