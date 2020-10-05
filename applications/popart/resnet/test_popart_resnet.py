# Copyright 2020 Graphcore Ltd.
import unittest
import os
import sys
import subprocess
import pytest
from tempfile import TemporaryDirectory
# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import parse_results_for_accuracy


def run_popart_resnet_training(**kwargs):
    with TemporaryDirectory() as tempdir:
        kwargs['--log-dir'] = tempdir
        cmd = ["python" + str(sys.version_info[0]), './resnet_main.py']
        # Flatten kwargs and convert to strings
        args = [str(item) for sublist in kwargs.items()
                for item in sublist if item != '']
        cmd.extend(args)
        out = subprocess.check_output(cmd, cwd=os.path.dirname(__file__)).decode("utf-8")
    return out


class TestPopARTResnetImageClassification(unittest.TestCase):
    """High-level integration tests training ResNets in popART"""

    @pytest.mark.ipus(4)
    @pytest.mark.category2
    def test_resnet8_bs4_4ipus(self):
        with TemporaryDirectory() as datadir:
            out = run_popart_resnet_training(
                **{
                    '--size': '8',
                    '--batch-size': 4,
                    '--norm-type': 'GROUP',
                    '--num-ipus': 4,
                    '--epochs': 5,
                    '--no-prng': '',
                    '--data-dir': datadir,
                    '--num-workers': 0
                }
            )
        expected_accuracy = [48.5, 53.6, 61.6, 64.6, 65.1]
        parse_results_for_accuracy(out, expected_accuracy, 4)

    @pytest.mark.ipus(4)
    @pytest.mark.category2
    def test_resnet8_bs4_4ipus_pipeline(self):
        with TemporaryDirectory() as datadir:
            out = run_popart_resnet_training(
                **{
                    '--size': '8',
                    '--batch-size': 4,
                    '--norm-type': 'GROUP',
                    '--num-ipus': 4,
                    '--epochs': 5,
                    '--pipeline': '',
                    '--no-prng': '',
                    '--data-dir': datadir,
                    '--num-workers': 0
                }
            )
        expected_accuracy = [42.6, 48.8, 63.7, 62.5, 61.0]
        parse_results_for_accuracy(out, expected_accuracy, 4)

    @pytest.mark.ipus(8)
    @pytest.mark.category2
    def test_resnet8_bs4_4ipus_pipeline_replication(self):
        with TemporaryDirectory() as datadir:
            out = run_popart_resnet_training(
                **{
                    '--size': '8',
                    '--batch-size': 8,
                    '--norm-type': 'GROUP',
                    '--num-ipus': 8,
                    '--epochs': 5,
                    '--pipeline': '',
                    '--replication-factor': 2,
                    '--no-prng': '',
                    '--data-dir': datadir,
                    '--num-workers': 0
                }
            )
        expected_accuracy = [42.25, 48.55, 59.4, 62.35, 61.0]
        parse_results_for_accuracy(out, expected_accuracy, 4)
