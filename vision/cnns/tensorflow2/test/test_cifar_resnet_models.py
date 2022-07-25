# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).absolute().parent.parent))
from model.model_factory import ModelFactory
from model.cifar_resnet_models import CifarResNet8, CifarResNet20, CifarResNet32, CifarResNet44, CifarResNet56


class CifarResNetTrainableParameters(unittest.TestCase):
    @staticmethod
    def get_num_of_trainable_weights(model):
        return np.sum([np.prod(layer.numpy().shape) for layer in model.trainable_weights])

    def test_cifar_resnet8_num_learnable_params(self):
        expected_num_trainable_weights = 75290
        num_trainable_weights = self.get_num_of_trainable_weights(model=CifarResNet8())
        self.assertEqual(num_trainable_weights, expected_num_trainable_weights)

    def test_cifar_resnet20_num_learnable_params(self):
        expected_num_trainable_weights = 269722
        num_trainable_weights = self.get_num_of_trainable_weights(model=CifarResNet20())
        self.assertEqual(num_trainable_weights, expected_num_trainable_weights)

    def test_cifar_resnet32_num_learnable_params(self):
        expected_num_trainable_weights = 464154
        num_trainable_weights = self.get_num_of_trainable_weights(model=CifarResNet32())
        self.assertEqual(num_trainable_weights, expected_num_trainable_weights)

    def test_cifar_resnet44_num_learnable_params(self):
        expected_num_trainable_weights = 658586
        num_trainable_weights = self.get_num_of_trainable_weights(model=CifarResNet44())
        self.assertEqual(num_trainable_weights, expected_num_trainable_weights)

    def test_cifar_resnet56_num_learnable_params(self):
        expected_num_trainable_weights = 853018
        num_trainable_weights = self.get_num_of_trainable_weights(model=CifarResNet56())
        self.assertEqual(num_trainable_weights, expected_num_trainable_weights)


class CreateModelTest(unittest.TestCase):
    def get_predictions_for_model(self, model_name: str):
        tf.random.set_seed(1)
        np.random.seed(0)
        image0 = np.zeros((1, 32, 32, 3))
        image1 = np.ones((1, 32, 32, 3)) * 10

        model = ModelFactory.create_model(model_name=model_name,
                                          input_shape=(32, 32, 3),
                                          classes=2)
        image0_preds = model.predict(image0)[0]
        image1_preds = model.predict(image1)[0]

        tf.random.set_seed(None)
        np.random.seed(None)

        return (image0_preds, image1_preds)

    def test_cifar_resnet8_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='cifar_resnet8')
        expected_image0_preds = np.allclose(image0_preds, [0.5, 0.5])
        expected_image1_preds = np.allclose(image1_preds, [0.00017077064, 0.99982917])
        self.assertTrue(expected_image0_preds, f'{image0_preds} != [0.5, 0.5]')
        self.assertTrue(expected_image1_preds, f'{image1_preds} != [0.00017077064, 0.99982917]')

    def test_cifar_resnet20_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='cifar_resnet20')
        expected_image0_preds = np.array_equal(image0_preds, [0.5, 0.5])
        expected_image1_preds = np.allclose(image1_preds, [0, 1])
        self.assertTrue(expected_image0_preds, f'{image0_preds} != [0.5, 0.5]')
        self.assertTrue(expected_image1_preds, f'{image1_preds} != [0.10094339, 0.8990566]')

    def test_cifar_resnet32_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='cifar_resnet32')
        expected_image0_preds = np.array_equal(image0_preds, [0.5, 0.5])
        expected_image1_preds = np.allclose(image1_preds, [1, 0])
        self.assertTrue(expected_image0_preds, f'{image0_preds} != [0.5, 0.5]')
        self.assertTrue(expected_image1_preds, f'{image1_preds} != [1, 0]')

    def test_cifar_resnet44_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='cifar_resnet44')
        expected_image0_preds = np.array_equal(image0_preds, [0.5, 0.5])
        expected_image1_preds = np.allclose(image1_preds, [1.0, 0.0])
        self.assertTrue(expected_image0_preds, f'{image0_preds} != [0.5, 0.5]')
        self.assertTrue(expected_image1_preds, f'{image1_preds} != [1.0, 0.0]')

    def test_cifar_resnet56_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='cifar_resnet56')
        expected_image0_preds = np.array_equal(image0_preds, [0.5, 0.5])
        expected_image1_preds = np.allclose(image1_preds, [1.0, 0.0])
        self.assertTrue(expected_image0_preds, f'{image0_preds} != [0.5, 0.5]')
        self.assertTrue(expected_image1_preds, f'{image1_preds} != [1.0, 0.0]')
