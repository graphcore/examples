# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest

from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestComplete(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_run_default_ipu(self):
        # Check default params
        self.run_command("python tuto_data_loading.py",
                         working_path,
                         "IPU throughput")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_run_synthetic_ipu(self):
        # Check synthetic data params
        self.run_command("python tuto_data_loading.py --synthetic-data",
                         working_path,
                         "IPU throughput")

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_run_replication_ipu(self):
        # Check replication
        self.run_command("python tuto_data_loading.py --replicas 2",
                         working_path,
                         "IPU throughput")

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_run_replication_synthetic_ipu(self):
        # Check synthetic data with replication
        self.run_command("python tuto_data_loading.py --replicas 2 --synthetic-data",
                         working_path,
                         "IPU throughput")
