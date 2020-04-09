# Copyright 2020 Graphcore Ltd.
from pathlib import Path

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import SubProcessChecker

working_path = Path(__file__).parent


class TestReadmeCommands(SubProcessChecker):

    def setUp(self):
        self.run_command("sh get_data.sh",
                         working_path,
                         ["Unpacking returns_and_features_for_mcmc.tar.gz"])

    def test_mcmc_model(self):
        self.run_command("python3 mcmc_tfp.py",
                         working_path,
                         ["MCMC sampling example with TensorFlow Probability",
                          "Warming up...",
                          "Completed in"])
