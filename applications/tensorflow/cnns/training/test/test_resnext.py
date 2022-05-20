# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

"""
Tests covering ResNeXt training.
"""

import pytest
import unittest
import statistics
import sys
from pathlib import Path
from examples_tests.test_util import SubProcessChecker

sys.path.append(str(Path(__file__).absolute().parent.parent))

from test_common import get_csv, run_train, cifar10_data_dir


@pytest.mark.ipus(1)
class TestCifar10ResNeXtTraining(SubProcessChecker):
    """Testing some basic training parameters"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--epochs': 10,
                '--model': "resnext",
                '--model-size': 29,
                '--micro-batch-size': 8,
                '--warmup-epochs': 0,
                '--learning-rate-decay': '0.1',
                '--learning-rate-schedule': '0.5,0.75,0.875'})
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

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


@pytest.mark.ipus(1)
class TestCifar10ResNeXtFullTraining(SubProcessChecker):
    """Fast training of Cifar-10 to good accuracy"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--epochs': 50,
                '--model': "resnext",
                '--model-size': 29,
                '--micro-batch-size': 4,
                '--warmup-epochs': 2,
                '--lr-schedule': 'cosine',
                '--label-smoothing': '0.05',
                '--base-learning-rate-exponent': -5,
                '--precision': '16.32'})
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

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


@pytest.mark.ipus(2)
class TestPipelineResNeXt14(SubProcessChecker):
    """A simple pipelined model"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--iterations': 500,
                '--dataset': 'imagenet',
                '--model': 'resnext',
                '--model-size': 14,
                '--generated-data': '',
                '--shards': 2,
                '--pipeline': '',
                '--gradient-accumulation-count': 128,
                '--micro-batch-size': 1,
                '--no-validation': '',
                '--pipeline-splits': 'stage2/unit1/relu',
                '--fused-preprocessing': ''})
        self.out = out
        self.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_iterations_completed
        self.assertEqual(self.training['iteration'][-1], 500)

        # test_number_of_parameters
        self.assertTrue('9415016' in self.out)


@pytest.mark.ipus(2)
class TestReplicatedTraining(SubProcessChecker):
    """Using replicas for data parallelism"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--model': 'resnext',
                '--lr-schedule': 'stepped',
                '--model-size': 29,
                '--micro-batch-size': 4,
                '--learning-rate-decay': 0.5,
                '--learning-rate-schedule': '0.5,0.9',
                '--epochs': 20,
                '--replicas': 2})
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 85)
        self.assertLess(final_acc, 95)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 20)


@pytest.mark.ipus(1)
class TestLotsOfOptions(SubProcessChecker):
    """Testing lots of other options to check they are still available"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--dataset': 'cifar-10',
                '--model': 'resnext',
                '--epochs': 2,
                '--model-size': 29,
                '--micro-batch-size': 4,
                '--batch-norm': '',
                '--pipeline-num-parallel': 8,
                '--generated-data': '',
                '--base-learning-rate-exponent': -4,
                '--precision': '32.32',
                '--seed': 1234,
                '--warmup-epochs': 0,
                '--stochastic-rounding': 'OFF',
                '--batches-per-step': 100})
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

    # We're mostly just testing that training still runs with all the above options.
    def test_results(self):
        # test_learning_rate
        self.assertEqual(self.training['lr'][0], 0.25)

        # test_epoch
        self.assertEqual(int(self.validation['epoch'][-1] + 0.5), 2)
