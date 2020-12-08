# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestReadmeCommands(SubProcessChecker):

    def test_download_data_and_run_readme_example(self):
        self.run_command("sh get_data.sh",
                         working_path,
                         ["Unpacking netflix_data.tar.gz"])
        self.run_command("python3 autoencoder_main.py --epochs 1 --training-data-file netflix_data/3m_train.txt"
                         " --validation-data-file netflix_data/3m_valid.txt --size 128",
                         working_path,
                         ["Loading training data", "Loading evaluation data", "Running validation..."])
