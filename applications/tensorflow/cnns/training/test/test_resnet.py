# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Tests covering various CNN training options using the ResNet model.
"""

import statistics
import unittest
import pytest
import sys
from pathlib import Path
from examples_tests.test_util import SubProcessChecker

sys.path.append(str(Path(__file__).absolute().parent.parent))

from test_common import get_csv, run_train, cifar10_data_dir


class TestBasicFunctionality(SubProcessChecker):
    """ Test that the help option works"""

    def test_help(self):
        help_out = run_train(
            self,
            **{'--help': ''})
        self.assertNotEqual(help_out.find("usage: train.py"), -1)


@pytest.mark.ipus(1)
class TestMisc(SubProcessChecker):
    """Some miscellaneous options"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--name-suffix': 'penguin',
                '--log-dir': 'logs/walrus',
                '--iterations': 10,
                '--batches-per-step': 10})
        self.logdir = None
        for line in out.split('\n'):
            if line.find('Saving to ') != -1:
                self.logdir = line[11:]
                break
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_logdir
        self.assertEqual(self.logdir[:12], 'logs/walrus/')
        # test_name_suffix
        self.assertNotEqual(self.logdir.find('penguin'), -1)


@pytest.mark.ipus(1)
class TestCifar10Training(SubProcessChecker):
    """Testing some basic training parameters"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--epochs': 10,
                '--warmup-epochs': 0,
                '--learning-rate-decay': '0.1',
                '--learning-rate-schedule': '0.5,0.75,0.875'})
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test_final_validation_accuracy
        final_acc = self.validation['val_acc'][-1]
        self.assertGreater(final_acc, 80)
        self.assertLess(final_acc, 87)

        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 80)
        self.assertLess(final_acc, 87)

        # test_learning_rates
        self.assertEqual(self.training['lr'][0], 0.5)
        self.assertEqual(self.training['lr'][-1], 0.0005)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 10)


@pytest.mark.ipus(1)
class TestCifar10FullTraining(SubProcessChecker):
    """Fast training of Cifar-10 to good accuracy"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--epochs': 50,
                '--micro-batch-size': 48,
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
        self.assertGreater(final_acc, 89.0)
        self.assertLess(final_acc, 91.0)

        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 96)
        self.assertLess(final_acc, 98)

        # test_final_loss
        self.assertLess(self.training['loss_batch'][-1], 0.45)
        self.assertGreater(self.training['loss_batch'][-1], 0.35)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 50)


@pytest.mark.ipus(1)
class TestResNet50SingleIPUTraining(SubProcessChecker):
    """ResNet-50 example on a single IPU.
        This is differs from the command line in the README:
        here we are testing with generated random data and only 10 iterations.
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--generated-data': '',
                '--dataset': 'ImageNet',
                '--model-size': 50,
                '--micro-batch-size': 1,
                '--available-memory-proportion': 0.1,
                '--iterations': 10,
                '--batches-per-step': 10})
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.ipus(2)
class TestResNet50Pipelining2IPUs(SubProcessChecker):
    """Pipelined ResNet-50 from the README but with generated random data
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
                '--model-size': 50,
                '--shards': 2,
                '--pipeline': '',
                '--gradient-accumulation-count': 256,
                '--micro-batch-size': 2,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--available-memory-proportion': 0.1,
                '--pipeline-splits': 'stage3/unit2/relu'})
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)
        self.assertGreater(self.training['loss_batch'][-1], 0)


@pytest.mark.ipus(4)
class TestResNet50Pipelining2IPUs2Replicas(SubProcessChecker):
    """Pipelined and replicated ResNet-50 from the README but with generated random
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
                '--model-size': 50,
                '--shards': 2,
                '--replicas': 2,
                '--pipeline': '',
                '--gradient-accumulation-count': 128,
                '--pipeline-schedule': 'Grouped',
                '--micro-batch-size': 2,
                '--no-validation': '',
                '--enable-recomputation': '',
                '--available-memory-proportion': 0.1,
                '--pipeline-splits': 'stage3/unit1/relu'})
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)
        self.assertGreater(self.training['loss_batch'][-1], 0)


@pytest.mark.ipus(2)
class TestReplicatedTraining(SubProcessChecker):
    """Using replicas for data parallelism"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--model': 'resnet',
                '--lr-schedule': 'stepped',
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
                '--epochs': 10,
                '--model-size': 14,
                '--batch-norm': '',
                '--pipeline-num-parallel': 8,
                '--generated-data': '',
                '--micro-batch-size': 16,
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
        self.assertEqual(self.training['lr'][0], 1.0)

        # test_epoch
        self.assertEqual(int(self.validation['epoch'][-1] + 0.5), 10)


@pytest.mark.ipus(1)
class TestConfig(SubProcessChecker):
    """Testing lots of other options to check they are still available"""

    def setUp(self):
        out = run_train(
            self,
            **{
                '--config': 'mk2_resnet8_test',
                '--data-dir': cifar10_data_dir})
        self.training = get_csv(out, 'training.csv')

    def test_results(self):
        # test the cmd line arg overrode config arg
        self.assertEqual(int(self.training['epoch'][-1]), 10)


@pytest.mark.ipus(2)
@pytest.mark.ipu_version("ipu2")
class TestResNet50RecomputeDbnTraining(SubProcessChecker):
    """ResNet-50 example on two IPUs with distributed batch norm and recompute.
    """

    def setUp(self):
        out = run_train(
            self,
            **{
                '--generated-data': '',
                '--dataset': 'ImageNet',
                '--model-size': 50,
                '--micro-batch-size': 8,
                '--available-memory-proportion': 0.1,
                '--iterations': 10,
                '--BN-span': 2,
                '--internal-exchange-optimisation-target': 'memory',
                '--pipeline': '',
                '--gradient-accumulation-count': 2,
                '--pipeline-schedule': 'Sequential',
                '--enable-recomputation': '',
                '--pipeline-splits': 'stage1/unit1/relu',
                '--eight-bit': '',
                '--replicas': 2,
                '--enable-half-partials': '',
                '--disable-variable-offloading': '',
                '--batch-norm': '',
                '--normalise-input': ''})
        self.validation = get_csv(out, 'validation.csv')
        self.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 500)
