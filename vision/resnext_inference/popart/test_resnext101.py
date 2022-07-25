# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestReadmeCommands(SubProcessChecker):

    def test_run_readme_example(self):
        self.run_command("python3 get_model.py --micro-batch-size 6",
                         working_path,
                         "Converting model to batch size ")
        self.run_command("python resnext_inference_launch.py --batch_size 48 --num_ipus 8 --synthetic",
                         working_path,
                         r"All processes finished with exit codes: \[0, 0, 0, 0, 0, 0, 0, 0\]")
