# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Tests covering various CNN training options using the EfficientNet-B0 model.
"""

import unittest
import pytest
import sys
from pathlib import Path
from examples_tests.test_util import SubProcessChecker

sys.path.append(str(Path(__file__).absolute().parent.parent))

from test_common import get_csv, run_train


class TestBasicFunctionality(SubProcessChecker):
    """ Test that the help option works"""
    def test_help(self):
        help_out = run_train(self, **{'--model': 'efficientnet', '--help': ''})
        self.assertNotEqual(help_out.find("EfficientNet:"), -1)


@pytest.mark.ipus(2)
class TestEfficientNetB0Pipelining2IPUs(SubProcessChecker):
    """EfficientNet-B0 example over 2 IPUs.
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--generated-data': '',
                '--dataset': 'ImageNet',
                '--model': 'efficientnet',
                '--shards': 2,
                '--pipeline': '',
                '--gradient-accumulation-count': 256,
                '--micro-batch-size': 2,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--available-memory-proportion': 0.2,
                '--iterations': 10,
                '--pipeline-splits': 'block3b',
                '--fused-preprocessing': ''})
        self.out = out
        self.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_iterations_completed
        self.assertEqual(self.training['iteration'][-1], 500)
        # test_number_of_parameters
        self.assertTrue('5288836' in self.out)
        # test_overall_batch_size
        self.assertTrue("Batch Size: 512" in self.out)


@pytest.mark.ipus(2)
class TestModifiedEfficientNetB0Pipelining2IPUs(SubProcessChecker):
    """Pipelined modified EfficientNet-B0 with generated random data
        and only 10 iterations.
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--iterations': 10,
                '--batches-per-step': 10,
                '--dataset': 'imagenet',
                '--generated-data': '',
                '--model': 'EfficientNet',
                '--model-size': 'B0',
                '--shards': 2,
                '--pipeline': '',
                '--gradient-accumulation-count': 128,
                '--micro-batch-size': 4,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--pipeline-schedule': 'Grouped',
                '--group-dim': 16,
                '--expand-ratio': 4,
                '--pipeline-splits': 'block3b'})
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.ipus(4)
class TestEfficientNetB0Pipelining2IPUs2Replicas(SubProcessChecker):
    """Pipelined and replicated modified EfficientNet with generated random
       data and only 10 iterations.
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--iterations': 10,
                '--batches-per-step': 10,
                '--dataset': 'imagenet',
                '--generated-data': '',
                '--model': 'EfficientNet',
                '--model-size': 'B0',
                '--shards': 2,
                '--replicas': 2,
                '--pipeline': '',
                '--gradient-accumulation-count': 128,
                '--pipeline-schedule': 'Grouped',
                '--micro-batch-size': 2,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--group-dim': 16,
                '--expand-ratio': 4,
                '--use-relu': '',
                '--available-memory-proportion': 0.2,
                '--pipeline-splits': 'block3b'})
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.ipus(1)
class TestEfficientNetCifar(SubProcessChecker):
    """EfficientNet for CIFAR datasets
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--iterations': 100,
                '--batches-per-step': 10,
                '--dataset': 'cifar-10',
                '--generated-data': '',
                '--model': 'EfficientNet',
                '--model-size': 'cifar',
                '--micro-batch-size': 10,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--group-dim': 16,
                '--expand-ratio': 4})
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 100)
