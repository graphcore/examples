# Copyright 2019 Graphcore Ltd.
import inspect
import unittest
import os
import sys
import subprocess
from contextlib import contextmanager

import tests.test_util as tu


def run_popart_resnet_training(**kwargs):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = ["python" + str(sys.version_info[0]), './resnet_main.py']
    # Flatten kwargs and convert to strings
    args = [str(item) for sublist in kwargs.items()
            for item in sublist if item != '']
    cmd.extend(args)
    out = subprocess.check_output(cmd, cwd=cwd).decode("utf-8")
    return out


class TestPopARTResnetImageClassification(unittest.TestCase):
    """High-level integration tests training ResNets in popART"""

    @classmethod
    def setUpClass(cls):
        pass

    def test_resnet8_bs4_4ipus(self):
        out = run_popart_resnet_training(**{'--size': 8,
                                            '--batch-size': 4,
                                            '--norm-type': 'GROUP',
                                            '--num-ipus': 4,
                                            '--epochs': 5,
                                            '--no-prng': '',
                                            '--data-dir': './',
                                            '--num-workers': 0})
        expected_accuracy = [41.0, 53.6, 59.0, 62.0, 62.7]
        tu.parse_results_for_accuracy(out, expected_accuracy, 2)

    def test_resnet8_bs4_4ipus_pipeline(self):
        out = run_popart_resnet_training(**{'--size': 8,
                                            '--batch-size': 4,
                                            '--norm-type': 'GROUP',
                                            '--num-ipus': 4,
                                            '--epochs': 5,
                                            '--pipeline': '',
                                            '--no-prng': '',
                                            '--data-dir': './',
                                            '--num-workers': 0})
        expected_accuracy = [42.2, 53.9, 60.7, 63.9, 65.6]
        tu.parse_results_for_accuracy(out, expected_accuracy, 2)
