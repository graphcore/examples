# Copyright 2019 Graphcore Ltd.
"""
Tests covering SqueezeNet training.
"""
import pytest
import unittest
import statistics

from test_common import get_csv, run_train, cifar10_data_dir


@pytest.mark.category2
@pytest.mark.ipus(1)
class TestCifar10SqueezeNetTraining(unittest.TestCase):
    """Testing some basic training parameters"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--model': "squeezenet",
                           '--epochs': 10,
                           '--use-bypass': '',
                           '--poly-lr-initial-lr': 0.1,
                           '--poly-lr-end-lr': 0.0001,
                           '--lr-schedule': "polynomial_decay_lr"})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_final_validation_accuracy(self):
        final_acc = self.validation['val_acc'][-1]
        self.assertGreater(final_acc, 57)
        self.assertLess(final_acc, 67)

    def test_final_training_accuracy(self):
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 55)
        self.assertLess(final_acc, 65)

    def test_learning_rates(self):
        self.assertEqual(self.training['lr'][0], 0.1)
        self.assertAlmostEqual(self.training['lr'][-1], 0.0001, places=3)

    def test_epochs_completed(self):
        self.assertEqual(round(self.training['epoch'][-1]), 10)


@pytest.mark.category3
@pytest.mark.ipus(1)
class TestCifar10SqueezeNetFullTraining(unittest.TestCase):
    """Fast training of Cifar-10 to good accuracy"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--epochs': 400,
                           '--model': "squeezenet",
                           '--use-bypass': '',
                           '--lr-schedule': 'polynomial_decay_lr',
                           '--label-smoothing': '0.05',
                           '--poly-lr-initial-lr': 0.1,
                           '--poly-lr-end-lr': 0.0001,
                           '--precision': '16.32'})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_final_validation_accuracy(self):
        final_acc = statistics.median(self.validation['val_acc'][-3:-1])
        self.assertGreater(final_acc, 83.0)
        self.assertLess(final_acc, 87.0)

    def test_final_training_accuracy(self):
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 96)
        self.assertLess(final_acc, 99)

    def test_final_loss(self):
        self.assertLess(self.training['loss_batch'][-1], 0.45)
        self.assertGreater(self.training['loss_batch'][-1], 0.35)

    def test_epochs_completed(self):
        self.assertEqual(round(self.training['epoch'][-1]), 399)
