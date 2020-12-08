# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

"""
Tests covering ResNeXt training.
"""

import pytest
import unittest
import statistics
from test_common import get_csv, run_train, cifar10_data_dir


@pytest.mark.category2
@pytest.mark.ipus(1)
class TestCifar10ResNeXtTraining(unittest.TestCase):
    """Testing some basic training parameters"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--epochs': 10,
                           '--model': "resnext",
                           '--model-size': 29,
                           '--batch-size': 8,
                           '--warmup-epochs': 0,
                           '--learning-rate-decay': '0.1',
                           '--learning-rate-schedule': '0.5,0.75,0.875'})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_final_validation_accuracy
        final_acc = self.validation['val_acc'][-1]
        self.assertGreater(final_acc, 82)
        self.assertLess(final_acc, 89)

        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 82)
        self.assertLess(final_acc, 89)

        # test_learning_rates
        self.assertEqual(self.training['lr'][0], 0.125)
        self.assertEqual(self.training['lr'][-1], 0.000125)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 10)


@pytest.mark.category3
@pytest.mark.ipus(1)
class TestCifar10ResNeXtFullTraining(unittest.TestCase):
    """Fast training of Cifar-10 to good accuracy"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--epochs': 50,
                           '--model': "resnext",
                           '--model-size': 29,
                           '--batch-size': 4,
                           '--warmup-epochs': 2,
                           '--lr-schedule': 'cosine',
                           '--label-smoothing': '0.05',
                           '--base-learning-rate': -5,
                           '--precision': '16.32'})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_final_validation_accuracy
        final_acc = statistics.median(self.validation['val_acc'][-3:-1])
        self.assertGreater(final_acc, 91.0)
        self.assertLess(final_acc, 95.0)

        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 96)
        self.assertLess(final_acc, 100)

        # test_final_loss
        self.assertLess(self.training['loss_batch'][-1], 0.45)
        self.assertGreater(self.training['loss_batch'][-1], 0.35)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 50)


@pytest.mark.category2
@pytest.mark.ipus(2)
class TestShardedTraining(unittest.TestCase):
    """Example of sharding for model parallelism"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--synthetic-data': '',
                           '--dataset': 'ImageNet',
                           '--model': 'resnext',
                           '--model-size': 14,
                           '--shards': 2,
                           '--batch-size': 2,
                           '--iterations': '5000',
                           '--no-validation': ''})
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 5000)


@pytest.mark.category2
@pytest.mark.ipus(2)
class TestPipelineResNeXt14(unittest.TestCase):
    """A simple pipelined model"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 500,
                           '--dataset': 'imagenet',
                           '--model': 'resnext',
                           '--model-size': 14,
                           '--synthetic-data': '',
                           '--shards': 2,
                           '--pipeline': '',
                           '--gradient-accumulation-count': 128,
                           '--batch-size': 1,
                           '--no-validation': '',
                           '--pipeline-splits': 'b2/0/relu'})
        cls.out = out
        cls.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_iterations_completed
        self.assertEqual(self.training['iteration'][-1], 500)

        # test_number_of_parameters
        self.assertTrue('9411880' in self.out)


@pytest.mark.category2
@pytest.mark.ipus(2)
class TestReplicatedTraining(unittest.TestCase):
    """Using replicas for data parallelism"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--model': 'resnext',
                           '--lr-schedule': 'stepped',
                           '--model-size': 29,
                           '--batch-size': 4,
                           '--learning-rate-decay': 0.5,
                           '--learning-rate-schedule': '0.5,0.9',
                           '--epochs': 20,
                           '--replicas': 2})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 85)
        self.assertLess(final_acc, 95)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 20)


@pytest.mark.category2
@pytest.mark.ipus(1)
class TestLotsOfOptions(unittest.TestCase):
    """Testing lots of other options to check they are still available"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--dataset': 'cifar-10',
                           '--model': 'resnext',
                           '--epochs': 10,
                           '--model-size': 29,
                           '--batch-size': 4,
                           '--batch-norm': '',
                           '--pipeline-num-parallel': 8,
                           '--synthetic-data': '',
                           '--base-learning-rate': -4,
                           '--precision': '32.32',
                           '--seed': 1234,
                           '--warmup-epochs': 0,
                           '--no-stochastic-rounding': '',
                           '--batches-per-step': 100
                           })
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    # We're mostly just testing that training still runs with all the above options.
    def test_results(self):
        # test_learning_rate
        self.assertEqual(self.training['lr'][0], 0.25)

        # test_epoch
        self.assertEqual(int(self.validation['epoch'][-1] + 0.5), 10)
