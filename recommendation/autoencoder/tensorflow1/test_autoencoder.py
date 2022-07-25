# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestReadmeCommands(SubProcessChecker):

    def test_run_readme_example(self):
        self.run_command("python3 autoencoder_main.py",
                         working_path,
                         ["Generating random training data", "Generating random evaluation data", "Running validation..."])
