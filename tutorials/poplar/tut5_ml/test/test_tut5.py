# Copyright 2020 Graphcore Ltd.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestBuildAndRun(SubProcessChecker):

    def setUp(self):
        ''' Download the MNIST data, if necessary.
            Compile the tutorial code '''
        self.run_command("./get_mnist.sh", working_path,
                         ["Checking", "(MNIST data already downloaded)|Done"])
        self.run_command("make clean", working_path, [])
        self.run_command("make all", working_path, [])

    def tearDown(self):
        self.run_command("make clean", working_path, [])

    @pytest.mark.category1
    def test_run_ipu_model(self):
        ''' Check that the tutorial code runs on the IPU Model'''

        self.run_command("./regression-demo 1 5.0",
                         working_path,
                         ["Using the IPU Model", "Epoch", "100%"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_ipu_hardware(self):
        ''' Check that the tutorial code runs on the IPU hardware'''

        self.run_command("./regression-demo -IPU 1 5.0",
                         working_path,
                         ["Using the IPU", "Epoch", "100%"])
