# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import bisect
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ipu.ops import pipelining_ops
import logging
from typing import Optional

from .toy_model import ToyModel
from .resnet_models import ResNet18, ResNet34, ResNet50
from .cifar_resnet_models import CifarResNet8, CifarResNet20, CifarResNet32, CifarResNet44, CifarResNet56
from .model_editor import ModelEditor
from eight_bit_transfer import EightBitTransfer
from custom_exceptions import DimensionError
from .model_editor import ModelEditor
from normalization import batch_norm
import numpy as np


AVAILABLE_MODELS = {'resnet50': ResNet50,
                    'resnet34': ResNet34,
                    'resnet18': ResNet18,
                    'cifar_resnet8': CifarResNet8,
                    'cifar_resnet20': CifarResNet20,
                    'cifar_resnet32': CifarResNet32,
                    'cifar_resnet44': CifarResNet44,
                    'cifar_resnet56': CifarResNet56,
                    'toy_model': lambda weights, input_shape, classes: ToyModel(input_shape=input_shape, classes=classes)}


class ModelFactory:

    logger = logging.getLogger('model_factory')

    @staticmethod
    def create_model(model_name: str,
                     input_shape: tuple,
                     classes: int,
                     weights: Optional[str] = None,
                     accelerator_side_preprocessing_fn=None,
                     eight_bit_transfer: Optional[EightBitTransfer] = None):

        if model_name not in AVAILABLE_MODELS.keys():
            raise NameError(
                f'Model {model_name} is not supported. '
                f'Supported models: {list(AVAILABLE_MODELS.keys())}')

        model = AVAILABLE_MODELS[model_name](weights=weights,
                                             input_shape=input_shape,
                                             classes=classes)

        if 'cifar' not in model_name:
            model = replace_bn_layers(model)

        if accelerator_side_preprocessing_fn is not None:
            model = preappend_fn_as_lambda_layer(model, accelerator_side_preprocessing_fn,
                                                 name='preprocessing_layer')

        if eight_bit_transfer is not None:
            model = preappend_fn_as_lambda_layer(model, eight_bit_transfer.decompress,
                                                 name='eight_bit_transfer_decompression_layer')

        ModelFactory.logger.info(f'Created a {model_name} model')

        return model

    @staticmethod
    def configure_model(model: keras.Model, gradient_accumulation_count: int, pipeline_splits: list,
                        device_mapping: list, pipeline_schedule: str, available_memory_proportion: list, optimizer_state_offloading: bool = True):

        if pipeline_splits:
            model = ModelFactory.pipeline_model(model, pipeline_splits)
            pipeline_schedule = next(p for p in list(pipelining_ops.PipelineSchedule)
                                     if pipeline_schedule == str(p).split(".")[-1])

            if device_mapping:
                if len(device_mapping) != len(pipeline_splits) + 1:
                    raise DimensionError(
                        f'The number of device assignments {len(device_mapping)} is not equal to the number of pipeline splits + 1: {len(pipeline_splits) + 1}.')

                if len(set(device_mapping)) != max(device_mapping) + 1:
                    raise DimensionError(
                        f'The model is pipelined over {len(set(device_mapping))} different IPUs, but one or more stages are being assigned to IPU {max(device_mapping) + 1}')

            if len(available_memory_proportion) > 1:

                if len(available_memory_proportion) != 2 * (len(pipeline_splits) + 1):
                    raise DimensionError(
                        'Define a single global value of available memory proportion or two values per pipeline stage. '
                        f'There are {len(pipeline_splits) + 1} pipeline stages defined and {len(available_memory_proportion)} values of '
                        'available memory proportion')

                options = [pipelining_ops.PipelineStageOptions(convolution_options={"availableMemoryProportion": str(available_memory_proportion[2 * idx] / 100.0)},
                                                               matmul_options={"availableMemoryProportion": str(available_memory_proportion[2 * idx + 1] / 100.0)})
                           for idx in range(len(available_memory_proportion) // 2)]
                kwargs = {'forward_propagation_stages_poplar_options': options,
                          'backward_propagation_stages_poplar_options': options}
            else:
                kwargs = {}

            model.set_pipelining_options(gradient_accumulation_steps_per_replica=gradient_accumulation_count,
                                         pipeline_schedule=pipeline_schedule, device_mapping=device_mapping,
                                         offload_weight_update_variables=optimizer_state_offloading, **kwargs)

        else:
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation_count,
                                                    offload_weight_update_variables=optimizer_state_offloading)

        return model

    @staticmethod
    def pipeline_model(model: keras.Model, pipeline_splits: list):
        assignments = model.get_pipeline_stage_assignment()
        stage_id = 0

        for assignment in assignments:
            if stage_id < len(pipeline_splits) and assignment.layer.name == pipeline_splits[stage_id]:
                stage_id += 1
            assignment.pipeline_stage = stage_id

        if stage_id < len(pipeline_splits):
            layers_available = []
            for assignment in assignments:
                layers_available.append(assignment.layer.name)
            raise NameError(
                f'Layer {pipeline_splits[stage_id]} not present in the model or invalid layers\' order provided. Layers available: {layers_available}')

        model.set_pipeline_stage_assignment(assignments)

        return model

    @staticmethod
    def evaluate_splits(model: keras.Model, num_pipeline_stages: int = 2):
        dimension = 0
        reached_dim_per_layer = []
        layer_names = []
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                for r in range(len(weights)):
                    dimension += np.prod(weights[r].shape)
                    reached_dim_per_layer.append(dimension)
                    layer_names.append(layer.name)

        param_per_stage = dimension / num_pipeline_stages
        k = 1
        split_list = []
        while(k * param_per_stage < dimension):
            layer_name = layer_names[bisect.bisect_left(reached_dim_per_layer, k * param_per_stage)]
            ModelFactory.logger.info(f'Stage {k} reached for layer {layer_name}')
            split_list.append(layer_name)
            k += 1
        return split_list


def preappend_fn_as_lambda_layer(model, fn, name='preappended_lambda_layer'):

    def model_editor_fn(current_layer, sub_input):
        if isinstance(current_layer, tf.keras.layers.InputLayer):
            preprocessed_output = tf.keras.layers.Lambda(lambda x: fn(x),
                                                         name=name)(sub_input)
            return preprocessed_output

    model_editor = ModelEditor(model)

    return model_editor.update_model_with_func(model_editor_fn)


def replace_preprocess_layer_with_fn(model, fn):

    if fn is not None:
        def model_editor_fn(current_layer, sub_input):
            if current_layer.name == 'preprocessing_layer':
                preprocessed_output = tf.keras.layers.Lambda(lambda x: fn(x),
                                                             name='preprocessing_layer')(sub_input)
                return preprocessed_output

        model_editor = ModelEditor(model)

        return model_editor.update_model_with_func(model_editor_fn)
    else:
        return model


def replace_bn_layers(model):

    def model_editor_fn(current_layer, layer_input):
        if isinstance(current_layer, tf.keras.layers.BatchNormalization):
            output = batch_norm.BatchNormIPU(name=current_layer.name, axis=3, epsilon=1.001e-5)(layer_input)
            return output

    model_editor = ModelEditor(model)

    return model_editor.update_model_with_func(model_editor_fn)
