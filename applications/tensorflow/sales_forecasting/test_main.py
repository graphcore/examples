# Copyright 2020 Graphcore Ltd
import os
import pytest
from tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class Test(SubProcessChecker):
    """ Test the sales forecasting model """
    @classmethod
    def setUpClass(self):
        super(Test, self).setUpClass()

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_sales_forecasting_one_ipu(self):
        """ Test that the model runs on one IPU for one epoch """
        self.run_command("python main.py --use-synthetic-data --epochs 1 --mov-mean-window 0",
                         working_path,
                         ["Begin training loop", "Training:", r"epoch:\s+1", "Validation:", "Best RMSPE:"])

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_sales_forecasting_two_ipus(self):
        """ Test that the model runs on two IPUs for one epoch """
        self.run_command("python main.py --replication-factor 2 --multiprocessing"
                         " --use-synthetic-data --epochs 1 --mov-mean-window 0",
                         working_path,
                         ["Begin training loop", "Training:", r"epoch:\s+1", "Validation:", "Best RMSPE:"])
