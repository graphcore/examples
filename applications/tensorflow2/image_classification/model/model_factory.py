# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import bisect
from math import gamma
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import optimizers
import logging
from typing import Optional, List
import tensorflow_addons as tfa
from ipu_tensorflow_addons.keras.layers import GroupNormalization

from .toy_model import ToyModel, ToyModelBn
from .resnet_models import ResNet18, ResNet34, ResNet50
from .cifar_resnet_models import CifarResNet8, CifarResNet20, CifarResNet32, CifarResNet44, CifarResNet56
from .model_editor import ModelEditor
from eight_bit_transfer import EightBitTransfer
from custom_exceptions import DimensionError
from .model_editor import ModelEditor
from normalization import batch_norm
from utilities import verify_params_present


AVAILABLE_MODELS = {'resnet50': ResNet50,
                    'resnet34': ResNet34,
                    'resnet18': ResNet18,
                    'cifar_resnet8': CifarResNet8,
                    'cifar_resnet20': CifarResNet20,
                    'cifar_resnet32': CifarResNet32,
                    'cifar_resnet44': CifarResNet44,
                    'cifar_resnet56': CifarResNet56,
                    'toy_model': lambda weights, input_shape, classes: ToyModel(input_shape=input_shape, classes=classes),
                    'toy_model_bn': lambda weights, input_shape, classes: ToyModelBn(input_shape=input_shape, classes=classes)}


class ModelFactory:

    logger = logging.getLogger('model_factory')

    @staticmethod
    def create_model(model_name: str,
                     input_shape: tuple,
                     classes: int,
                     weights: Optional[str] = None,
                     norm_layer_params: dict = {'name': 'custom_batch_norm'},
                     accelerator_side_preprocessing_fn=None,
                     eight_bit_transfer: Optional[EightBitTransfer] = None):

        if model_name not in AVAILABLE_MODELS.keys():
            raise NameError(
                f'Model {model_name} is not supported. '
                f'Supported models: {list(AVAILABLE_MODELS.keys())}')

        model = AVAILABLE_MODELS[model_name](weights=weights,
                                             input_shape=input_shape,
                                             classes=classes)

        if norm_layer_params['name'] == 'group_norm':
            model = replace_bn_with_gn_layers(model, norm_layer_params)
        elif norm_layer_params['name'] == 'custom_batch_norm':
            model = replace_bn_with_custom_bn_layers(model)

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
                                         gradient_accumulation_reduction_method=optimizers.GradientAccumulationReductionMethod.RUNNING_MEAN,
                                         pipeline_schedule=pipeline_schedule, device_mapping=device_mapping,
                                         offload_weight_update_variables=optimizer_state_offloading, **kwargs)

        else:
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation_count,
                                                    gradient_accumulation_reduction_method=optimizers.GradientAccumulationReductionMethod.RUNNING_MEAN,
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

    @staticmethod
    def debug_layers(model, debug_layers_names: List[str]):

        debug_outfeed_queues = []

        if len(debug_layers_names) > 0:
            def model_editor_fn(current_layer, sub_input):
                if current_layer.name in debug_layers_names:
                    inputs_outfeed = ipu_outfeed_queue.IPUOutfeedQueue()
                    outputs_outfeed = ipu_outfeed_queue.IPUOutfeedQueue()
                    debug_outfeed_queues.append((f'{current_layer.name}_input', inputs_outfeed))
                    debug_outfeed_queues.append((f'{current_layer.name}_output', outputs_outfeed))
                    outputs = ModelFactory.LayerDebugger(current_layer, inputs_outfeed, outputs_outfeed)(sub_input)
                    return outputs

            model_editor = ModelEditor(model)
            model = model_editor.update_model_with_func(model_editor_fn, copy_weights=False)

        return model, debug_outfeed_queues

    class LayerDebugger(tf.keras.layers.Layer):
        def __init__(self, layer, inputs_outfeed, outputs_outfeed):
            super(ModelFactory.LayerDebugger, self).__init__(name=f'{layer.name}_debugger')
            self.layer = layer.from_config(layer.get_config())
            self.inputs_outfeed = inputs_outfeed
            self.outputs_outfeed = outputs_outfeed

        def build(self, input_shape):
            super().build(input_shape)
            self.layer.build(input_shape)

        def call(self, *args, **kwargs):
            self.inputs_outfeed.enqueue(args)
            outputs = self.layer(*args, **kwargs)
            self.outputs_outfeed.enqueue(outputs)
            return outputs

        def get_config(self):
            config = self.layer.get_config()
            config['inputs_outfeed'] = self.inputs_outfeed
            config['outputs_outfeed'] = self.outputs_outfeed
            config['layer_type'] = type(self.layer)
            return config

        @classmethod
        def from_config(cls, config):
            inputs_outfeed = config.pop('inputs_outfeed')
            outputs_outfeed = config.pop('outputs_outfeed')
            inner_layer_type = config.pop('layer_type')
            inner_layer = inner_layer_type(**config)
            return cls(inner_layer, inputs_outfeed, outputs_outfeed)


def preappend_fn_as_lambda_layer(model, fn, name='preappended_lambda_layer'):

    def model_editor_fn(current_layer, sub_input):
        if isinstance(current_layer, tf.keras.layers.InputLayer):
            preprocessed_output = tf.keras.layers.Lambda(lambda x: fn(x),
                                                         name=name)(sub_input)
            return preprocessed_output

    model_editor = ModelEditor(model)

    return model_editor.update_model_with_func(model_editor_fn, copy_weights=False)


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


def replace_bn_with_custom_bn_layers(model):

    def model_editor_fn(current_layer, layer_input):
        if isinstance(current_layer, tf.keras.layers.BatchNormalization):
            output = batch_norm.BatchNormIPU(name=current_layer.name,
                                             axis=current_layer.axis,
                                             gamma_initializer=current_layer.gamma_initializer,
                                             beta_initializer=current_layer.beta_initializer,
                                             epsilon=1.001e-5)(layer_input)
            return output

    model_editor = ModelEditor(model)

    return model_editor.update_model_with_func(model_editor_fn)


def replace_bn_with_gn_layers(model, params):

    verify_params_present(params=list(params.keys()),
                          expected_params=['channels_per_group', 'num_groups'],
                          object_name='group_norm_layers',
                          arg_name='--norm-layer',
                          all_or_any='any')

    if 'num_groups' in params.keys():
        if 'channels_per_group' in params.keys():
            raise ValueError('Both num_groups and channels_per_group cannot be specified at the same time '
                             'in defining group norm layers. Use only one of them.')
        num_groups = params['num_groups']
        channels_per_group = None
    else:
        num_groups = None
        channels_per_group = params['channels_per_group']

    def model_editor_fn(current_layer, layer_input):

        if isinstance(current_layer, tf.keras.layers.BatchNormalization):
            name = current_layer.name.replace('bn', 'gn')
            groups = num_groups or layer_input.shape[-1] // channels_per_group
            output = GroupNormalization(name=name,
                                        groups=groups,
                                        channels_axis=len(layer_input.shape) - 1,
                                        epsilon=1.001e-5,
                                        beta_initializer=current_layer.beta_initializer,
                                        gamma_initializer=current_layer.gamma_initializer)(layer_input)
            return output

    model_editor = ModelEditor(model)

    return model_editor.update_model_with_func(model_editor_fn)
