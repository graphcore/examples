# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

sys.path.append(str(Path(__file__).absolute().parent.parent))
from model.model_factory import ModelFactory
from model.toy_model import ToyModel
from model.resnet_models import ResNet50
from custom_exceptions import DimensionError


class ResNetVersionTest(unittest.TestCase):
    @staticmethod
    def get_num_of_trainable_weights(model):
        return np.sum([np.prod(layer.numpy().shape) for layer in model.trainable_weights])

    def test_resnet50_num_learnable_parameters(self):
        '''A test of whether ResNet50 implementations in TF1 and TF2 are alike.'''
        NUM_OF_LEARNABLE_PARAMETERS_TF1 = 25557032
        num_trainable_weights_tf2 = self.get_num_of_trainable_weights(model=ResNet50())
        assert num_trainable_weights_tf2 == NUM_OF_LEARNABLE_PARAMETERS_TF1


class UnsupportedModelTest(unittest.TestCase):
    def test_unsupported_model(self):
        with self.assertRaises(NameError):
            ModelFactory.create_model(model_name='foo',
                                      input_shape=(32, 32, 3),
                                      classes=2)


class InvalidStageNameTest(unittest.TestCase):
    def test_invalid_layer_name(self):
        train_strategy = ipu.ipu_strategy.IPUStrategy()
        with train_strategy.scope():
            model = ModelFactory.create_model(model_name='toy_model', weights=None, input_shape=(28, 28, 1), classes=10)
        with self.assertRaises(NameError):
            model = ModelFactory.configure_model(model=model, gradient_accumulation_count=1, pipeline_splits=[
                                                 'foo'], device_mapping=[], pipeline_schedule='Grouped',
                                                 available_memory_proportion=[])

    def test_invalid_layer_order(self):
        train_strategy = ipu.ipu_strategy.IPUStrategy()
        with train_strategy.scope():
            model = ModelFactory.create_model(model_name='toy_model', weights=None, input_shape=(28, 28, 1), classes=10)
        with self.assertRaises(NameError):
            model = ModelFactory.configure_model(model=model, gradient_accumulation_count=1, pipeline_splits=[
                                                 'conv2d_1', 'conv2d'], device_mapping=[], pipeline_schedule='Grouped',
                                                 available_memory_proportion=[])


class InvalidDeviceMappingTest(unittest.TestCase):
    def test_invalid_number_of_device_mapping(self):
        train_strategy = ipu.ipu_strategy.IPUStrategy()
        with train_strategy.scope():
            model = ModelFactory.create_model(model_name='toy_model', weights=None, input_shape=(28, 28, 1), classes=10)
        with self.assertRaises(DimensionError):
            model = ModelFactory.configure_model(model=model, gradient_accumulation_count=1, pipeline_splits=[
                                                 'conv2d_1', 'flatten'], device_mapping=[0, 1], pipeline_schedule='Grouped',
                                                 available_memory_proportion=[])

    def test_invalid_id_of_device_mapping(self):
        train_strategy = ipu.ipu_strategy.IPUStrategy()
        with train_strategy.scope():
            model = ModelFactory.create_model(model_name='toy_model', weights=None, input_shape=(28, 28, 1), classes=10)
        with self.assertRaises(DimensionError):
            model = ModelFactory.configure_model(model=model, gradient_accumulation_count=1, pipeline_splits=[
                                                 'conv2d_1', 'flatten'], device_mapping=[1, 2, 3], pipeline_schedule='Grouped',
                                                 available_memory_proportion=[])


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

    def test_resnet50_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='resnet50')
        assert (np.array_equal(image0_preds, [0.5, 0.5]))
        assert (np.allclose(image1_preds, [0.45950672, 0.54049337]))

    def test_resnet34_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='resnet34')
        assert (np.array_equal(image0_preds, [0.5, 0.5]))
        assert (np.allclose(image1_preds, [0.3269672, 0.67303276]))

    def test_resnet18_output(self):
        image0_preds, image1_preds = self.get_predictions_for_model(model_name='resnet18')
        assert (np.array_equal(image0_preds, [0.5, 0.5]))
        assert (np.allclose(image1_preds, [0.5110175, 0.48898253]))


class CreateToyModelTest(unittest.TestCase):
    def test_toy_model_prediction(self):
        tf.random.set_seed(1)
        model = ToyModel(input_shape=(32, 32, 3), classes=10)
        image_1 = np.ones((1, 32, 32, 3)) * 10

        assert (np.allclose(
            model.predict(image_1)[0], [0.08292384, 0.05735856, 0.27028584, 0.2666999, 0.02177826,
                                        0.01853362, 0.06498592, 0.04272136, 0.15957771, 0.015135]))
        tf.random.set_seed(None)


class CreateToyModelInFactory(unittest.TestCase):

    def test_toy_model_factory_prediction(self):
        tf.random.set_seed(1)
        model = ModelFactory.create_model(model_name='toy_model',
                                          weights=None,
                                          input_shape=(32, 32, 3),
                                          classes=10)
        image_1 = np.ones((1, 32, 32, 3)) * 10
        assert (np.allclose(
            model.predict(image_1)[0], [0.08292384, 0.05735856, 0.27028584, 0.2666999, 0.02177826,
                                        0.01853362, 0.06498592, 0.04272136, 0.15957771, 0.015135]))
        tf.random.set_seed(None)


class ConfigurePipelineTest(unittest.TestCase):
    def test_pipeline_split(self):

        def initial_model_1():
            model_input = keras.Input(shape=(32, 32, 3))
            model_output = keras.layers.MaxPooling2D(name='test_pipeline_split_layer1')(model_input)
            model_output_1 = keras.layers.Conv2D(
                filters=32, kernel_size=3, name='test_pipeline_split_layer2')(model_output)
            model_output_2 = keras.layers.Conv2D(
                filters=32, kernel_size=3, name='test_pipeline_split_layer3')(model_output)
            model_output = keras.layers.Add(name='test_pipeline_split_layer4')([model_output_1, model_output_2])
            model_output = keras.layers.Flatten(name='test_pipeline_split_layer5')(model_output)
            return keras.Model(model_input, model_output)

        def expected_model_1():
            model_input = keras.Input(shape=(32, 32, 3))
            with ipu.keras.PipelineStage(0):
                model_output = keras.layers.MaxPooling2D()(model_input)
                model_output_1 = keras.layers.Conv2D(filters=32, kernel_size=3)(model_output)
            with ipu.keras.PipelineStage(1):
                model_output_2 = keras.layers.Conv2D(filters=32, kernel_size=3)(model_output)
                model_output = keras.layers.Add()([model_output_1, model_output_2])
            with ipu.keras.PipelineStage(2):
                model_output = keras.layers.Flatten()(model_output)
            return keras.Model(model_input, model_output)

        train_strategy = ipu.ipu_strategy.IPUStrategy()
        with train_strategy.scope():
            model = initial_model_1()
            pipelined_model = ModelFactory.configure_model(model=model, gradient_accumulation_count=1, pipeline_splits=[
                                                           'test_pipeline_split_layer3', 'test_pipeline_split_layer5'],
                                                           device_mapping=[], pipeline_schedule='Grouped',
                                                           available_memory_proportion=[])

            expected_assignments = expected_model_1().get_pipeline_stage_assignment()
            pipelined_assignments = pipelined_model.get_pipeline_stage_assignment()

            for expected_assignment, pipelined_assignment in zip(expected_assignments, pipelined_assignments):
                assert(expected_assignment.layer.__class__.name == pipelined_assignment.layer.__class__.name)
                assert(expected_assignment.pipeline_stage == pipelined_assignment.pipeline_stage)
