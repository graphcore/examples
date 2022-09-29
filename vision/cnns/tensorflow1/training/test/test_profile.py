# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Tests covering various CNN training options using the EfficientNet-B0 model.
"""

import unittest
import pytest
import sys
import json
from pathlib import Path
from examples_tests.test_util import SubProcessChecker

sys.path.append(str(Path(__file__).absolute().parent.parent))

from test_common import get_csv, run_train, get_log_dir
import log as logging


@pytest.mark.ipus(1)
class TestProfile(SubProcessChecker):
    """EfficientNet for CIFAR datasets
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--iterations': 100,
                '--device-iterations': 10,
                '--dataset': 'cifar-10',
                '--generated-data': '',
                '--model': 'EfficientNet',
                '--model-size': 'cifar',
                '--micro-batch-size': 10,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--group-dim': 16,
                '--expand-ratio': 4,
                '--profile': ''})
        self.log_dir = get_log_dir(out)
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        """In profiling mode only one step is taken"""
        self.assertEqual(self.training['step'][-1], 1)

    def test_profile(self):
        opts = json.load(open(Path(self.log_dir) / "arguments.json"))
        assert [*Path(self.log_dir).rglob("profile.pop")], "No profile files found"
        logging.process_profile(opts)


@pytest.mark.ipus(1)
class TestCompilationProfile(SubProcessChecker):
    """EfficientNet for CIFAR datasets
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--iterations': 100,
                '--device-iterations': 10,
                '--dataset': 'cifar-10',
                '--generated-data': '',
                '--model': 'EfficientNet',
                '--model-size': 'cifar',
                '--micro-batch-size': 10,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--group-dim': 16,
                '--expand-ratio': 4,
                '--profile-compilation': ''})
        self.log_dir = get_log_dir(out)
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        """In profile-compilation mode execution is unaffected."""
        self.assertEqual(self.training['step'][-1], 10)

    def test_profile(self):
        opts = json.load(open(Path(self.log_dir) / "arguments.json"))
        assert [*Path(self.log_dir).rglob("profile.pop")], "No profile files found"
        logging.process_profile(opts)
