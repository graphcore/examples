# Copyright 2020 Graphcore Ltd.
import os

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class TestReadmeCommands(SubProcessChecker):

    def test_ssd_model(self):
        self.run_command("pip3 install h5py pillow",
                         working_path,
                         [])
        self.run_command("python ssd_model.py",
                         working_path,
                         ["Mean TFLOPs/S is"])
        self.run_command("python ssd_single_image.py",
                         working_path,
                         ["Done running inference."])
