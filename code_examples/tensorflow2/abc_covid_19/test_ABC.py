# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""Tests for the COVID-19 ABC algorithm"""
import pytest
from pathlib import Path
from examples_tests.test_util import SubProcessChecker


current_path = Path(__file__).parent


class AbcTest(SubProcessChecker):
    """Test simple command line executions"""

    @pytest.mark.category2
    @pytest.mark.ipus(1)
    def test_default(self):
        """Test the defaults"""
        self.run_command(
            "python ABC_IPU.py", current_path, ["Time per run"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_italy(self):
        self.run_command(
            "python ABC_IPU.py -cn Italy -t 2e5 -b 100 -s 1",
            current_path, ["Time per run"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_nz(self):
        self.run_command(
            'python ABC_IPU.py -cn New_Zealand -t 1e5 -b 100 -s 1',
            current_path, ["Time per run"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_usa(self):
        self.run_command(
            "python ABC_IPU.py -cn US -t 1e6 -b 100 -s 1",
            current_path, ["Time per run"])

    @pytest.mark.category1
    @pytest.mark.ipus(2)
    def test_replication(self):
        self.run_command(
            "python ABC_IPU.py -r 2 -b 100 -s 1",
            current_path, ["Time per run"])

    @pytest.mark.category2
    @pytest.mark.ipus(4)
    def test_readme(self):
        self.run_command(
            "python ABC_IPU.py --enqueue-chunk-size 10000 --tolerance 5e5 "
            "--n-samples-target 100 --n-samples-per-batch 400000 --country US"
            " --samples-filepath US_5e5_100.txt --replication-factor 4",
            current_path, ["Time per run"])
