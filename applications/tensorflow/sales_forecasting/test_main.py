# Copyright 2020 Graphcore Ltd
import os
from tempfile import TemporaryDirectory

import pytest

from examples_tests.test_util import SubProcessChecker


working_path = os.path.dirname(__file__)


class Test(SubProcessChecker):
    """ Test the sales forecasting model """
    @classmethod
    def setUpClass(self):
        super(Test, self).setUpClass()

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_sales_forecasting_one_ipu(self):
        """Test that the model runs on one IPU for one epoch."""
        with TemporaryDirectory() as temp_dir:
            self.run_command(
                (
                    "python3 main.py --use-synthetic-data --epochs 1"
                    f" --mov-mean-window 0 --log-dir {temp_dir}"
                ),
                working_path,
                [
                    "Begin training loop", "Training:", r"epoch:\s+1",
                    "Validation:", "Best RMSPE|no valid RMSPE results"
                ]
            )

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_sales_forecasting_two_ipus(self):
        """Test that the model runs when replicated over two IPUs for
           one epoch."""
        with TemporaryDirectory() as temp_dir:
            self.run_command(
                (
                    "python3 main.py --use-synthetic-data --epochs 1"
                    f" --mov-mean-window 0 --log-dir {temp_dir}"
                    " --replication-factor 2"
                ),
                working_path,
                [
                    "Begin training loop", "Training:", r"epoch:\s+1",
                    "Validation:", "Best RMSPE|no valid RMSPE results"
                ]
            )

    @pytest.mark.ipus(2)
    @pytest.mark.category1
    def test_sales_forecasting_multiprocessing(self):
        """Test that the model runs with multiprocessing enabled."""
        with TemporaryDirectory() as temp_dir:
            self.run_command(
                (
                    "python3 main.py --use-synthetic-data --epochs 1"
                    f" --mov-mean-window 0 --log-dir {temp_dir}"
                    " --multiprocessing"
                ),
                working_path,
                [
                    "Begin training loop", "Training:", r"epoch:\s+1",
                    "Validation:", "Best RMSPE|no valid RMSPE results"
                ]
            )
