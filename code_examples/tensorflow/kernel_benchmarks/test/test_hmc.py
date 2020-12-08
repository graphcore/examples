# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path

import pytest
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


class TestTensorFlowHamiltonianMonteCarloBenchmarks(SubProcessChecker):
    """High-level integration tests for TensorFlow Hamiltonian Monte-Carlo synthetic benchmarks"""

    @pytest.mark.category1
    def test_help(self):
        self.run_command("python3 hmc.py --help",
                         working_path,
                         "usage: hmc.py")

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_default(self):
        self.run_command("python3 hmc.py",
                         working_path,
                         [r"(\w+.\w+) hmc steps/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_multiple_steps(self):
        self.run_command("python3 hmc.py --steps 500",
                         working_path,
                         [r"(\w+.\w+) hmc steps/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_all_options(self):
        self.run_command("python3 hmc.py --steps 200 --hmc-steps 10000 --leapfrog-steps 10",
                         working_path,
                         [r"(\w+.\w+) hmc steps/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_report(self):
        self.run_command("python3 hmc.py --report",
                         working_path,
                         "IPU Timings")
