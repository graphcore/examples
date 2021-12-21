# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import pytest
import subprocess

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker


class TestReadmeCommands(SubProcessChecker):

    @pytest.mark.ipus(4)
    def test_edge_convolutional_network(self, args=""):
        """Run Edge Conditioned Graph Convolutional Neural Network on 133,000 molecules from the QM9 dataset
        """
        cmd = " python3 qm9_ipu.py"
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        build_dir = os.path.dirname(os.path.realpath(__file__))
        self.run_command(f"{cmd} {args}", build_dir, [r"Completed"], env=env)
