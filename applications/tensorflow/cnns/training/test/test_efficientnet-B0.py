# Copyright 2020 Graphcore Ltd.
"""
Tests covering various CNN training options using the EfficientNet-B0 model.
"""

import unittest
import pytest

from test_common import get_csv, parse_csv, run_train, cifar10_data_dir


@pytest.mark.category1
class TestBasicFunctionality(unittest.TestCase):
    """ Test that the help option works"""
    def test_help(self):
        help_out = run_train(**{'--model': 'efficientnet', '--help': ''})
        self.assertNotEqual(help_out.find("EfficientNet:"), -1)


@pytest.mark.category2
@pytest.mark.ipus(2)
class TestEfficientNetB0Pipelineing2IPUs(unittest.TestCase):
    """EfficientNet-B0 example over 2 IPUs.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--synthetic-data': '',
                           '--dataset': 'ImageNet',
                           '--model': 'efficientnet',
                           '--shards': 2,
                           '--pipeline-depth': 256,
                           '--batch-size': 2,
                           '--no-validation': '',
                           '--xla-recompute': '',
                           '--available-memory-proportion': 0.2,
                           '--iterations': 10,
                           '--pipeline-splits': 'block3b'})
        cls.out = out
        cls.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_iterations_completed
        self.assertEqual(self.training['iteration'][-1], 500)
        # test_number_of_parameters
        self.assertTrue('5288548' in self.out)
        # test_overall_batch_size
        self.assertTrue("Batch Size: 512" in self.out)


@pytest.mark.category2
@pytest.mark.ipus(2)
class TestModifiedEfficientNetB0Pipelining2IPUs(unittest.TestCase):
    """Pipelined modified EfficientNet-B0 with synthetic data
        and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 10,
                           '--batches-per-step': 10,
                           '--dataset': 'imagenet',
                           '--synthetic-data': '',
                           '--model': 'EfficientNet',
                           '--model-size': 'B0',
                           '--shards': 2,
                           '--pipeline-depth': 128,
                           '--batch-size': 4,
                           '--no-validation': '',
                           '--xla-recompute': '',
                           '--pipeline-schedule': 'Grouped',
                           '--group-dim': 16,
                           '--expand-ratio': 4,
                           '--pipeline-splits': 'block3b'})
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.category2
@pytest.mark.ipus(4)
class TestEfficientNetB0Pipelining2IPUs2Replicas(unittest.TestCase):
    """Pipelined and replicated modified EfficientNet with synthetic
       data and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 10,
                           '--batches-per-step': 10,
                           '--dataset': 'imagenet',
                           '--synthetic-data': '',
                           '--model': 'EfficientNet',
                           '--model-size': 'B0',
                           '--shards': 2,
                           '--replicas': 2,
                           '--pipeline-depth': 128,
                           '--pipeline-schedule': 'Grouped',
                           '--batch-size': 2,
                           '--no-validation': '',
                           '--xla-recompute': '',
                           '--group-dim': 16,
                           '--expand-ratio': 4,
                           '--use-relu': '',
                           '--available-memory-proportion': 0.2,
                           '--pipeline-splits': 'block3b'})
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.category2
@pytest.mark.ipus(1)
class TestEfficientNetCifar(unittest.TestCase):
    """EfficientNet for CIFAR datasets
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 100,
                           '--batches-per-step': 10,
                           '--dataset': 'cifar-10',
                           '--synthetic-data': '',
                           '--model': 'EfficientNet',
                           '--model-size': 'cifar',
                           '--batch-size': 10,
                           '--no-validation': '',
                           '--xla-recompute': '',
                           '--group-dim': 16,
                           '--expand-ratio': 4})
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 100)
