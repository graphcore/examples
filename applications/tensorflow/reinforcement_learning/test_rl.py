# Copyright 2020 Graphcore Ltd.
import os

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestReadmeCommands(SubProcessChecker):

    def test_reinforcement_learning_model(self):
        self.run_command("python3 rl_benchmark.py --batch_size 8 --time_steps 16 --num_ipus 8",
                         working_path,
                         [r"Average (\w+.\w+) items/sec"])
