# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from core import common
from tensorflow.python import ipu
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import StochasticRoundingBehaviour


def get_ipu_config(ipu_id=-1,
                   num_ipus_required=0,
                   fp_exceptions=True,
                   stochastic_rounding=True,
                   xla_recompute=False,
                   available_memory_proportion=None,
                   max_cross_replica_buffer_size=0,
                   scheduler_selection='',
                   compile_only=False,
                   partials_type="half"):
    """Builds ipu_options"""
    config = ipu.config.IPUConfig()
    config.selection_order = utils.SelectionOrder.AUTO
    config.optimizations.maximum_cross_replica_sum_buffer_size = max_cross_replica_buffer_size

    if ipu_id >= 0:
        config.select_ipus = ipu_id
    else:
        config.auto_select_ipus = num_ipus_required

    if available_memory_proportion is not None:
        config.convolutions.poplar_options['availableMemoryProportion'] = str(available_memory_proportion)
        config.matmuls.poplar_options['availableMemoryProportion'] = str(available_memory_proportion)
    config.convolutions.poplar_options['partialsType'] = partials_type
    config.matmuls.poplar_options['partialsType'] = partials_type

    config.norms.use_stable_statistics = True

    config.allow_recompute = xla_recompute

    if compile_only:
        config.device_connection.version = 'ipu2'
        config.device_connection.enable_remote_buffers = True
        config.device_connection.type = utils.DeviceConnectionType.NEVER

    config.floating_point_behaviour.inv = fp_exceptions
    config.floating_point_behaviour.div0 = fp_exceptions
    config.floating_point_behaviour.oflo = fp_exceptions
    config.floating_point_behaviour.esr = StochasticRoundingBehaviour.from_bool(stochastic_rounding)

    config.floating_point_behaviour.nanoo = fp_exceptions
    return config


def get_var_list(func):
    """
        get variable names of func, exclude "self" if there is
    """
    func_code = func.__code__
    var_list = func_code.co_varnames[:func_code.co_argcount]
    var_list = [var for var in var_list if var != 'self']
    return var_list


def stage_wrapper(layer_list, needed_vars, stage_input_names):
    """a wrapper that generate stage function dynamically

    Args:
        layer_list: a list of model layer functions,
            layer's output must be a dictionary so that stage_function will know which param is needed by remaining layers
        needed_values: list of string, values name that will be useful for rest stages
        stage_input_names: stage function need to output a list of tensors,
            so we need this additional list to keep the name for each tensor.
            stage_input_names will be updated at the end of each stage.
            stage_input_names will be in same order with needed_vars.

    Returns:
        a function that takes needed_vars concatenated and some key_word_args as it's inputs,
        sequentially call functions in "layer_list",
        and return a list of tensors that occur in "needed_vars" collected from each layer's output.
    """

    def stage_func(*args, **kwargs):
        """
        Args:
            args: can be from "inputs" of pipeline function or previous stage,
                if dataset is a list (instead of a dictionary), then it's values is also passed input args,
                that way,stage_input_names need to contain names for each value in dataset
            kwargs: can be from dataset, if dataset is a dictionary.
        """
        result = kwargs

        args = list(args)
        # args come from "inputs" argument of "pipeline" function
        result.update(zip(stage_input_names, args))

        for func in layer_list:
            var_list = get_var_list(func)
            outputs = func(**{name: result[name]
                              for name in var_list if name in result})
            # assume outputs to be a bectionary
            assert isinstance(outputs, dict)
            result.update(outputs)
        # only return needed vlaues
        # if a value of outputs is None
        # it means this value is pointless
        # so remove it
        result = OrderedDict([(key, result[key])
                              for key in needed_vars if key in result.keys() and result[key] is not None])
        # stage function can't return dictionary, so keep key in additional list
        # clear this list for use by next stage
        stage_input_names.clear()
        # now "stage_input_names" contains output name for current stage
        # and  at the same time, it's the input_name for next stage
        stage_input_names.extend(result.keys())
        return [result[key] for key in stage_input_names]
    return stage_func


def stages_constructor(stages_list, input_names, output_vars):
    """construct compuational_stages for pipeline

    Args:
        stages_list: list of list of layer functions.
            each list in stages_list represent a stage,
            function layers must output a dictionary so that this funciton can know name of it's output values
        input_names: appended inputs name list,
            if values are passed by "inputs" argument of pipeline function,
            this list will contain name of each value of it in the sequence of "inputs" value.
            if dataset is a list(instead of a dictionary), name for each value of it need to be
            appended after name of "inputs"
        output_vars: values output by last stage (used by optimizer)

    Returns:
        a list of stage functions
    """
    needed_vars = output_vars
    computational_stages = []
    # input names for args of a stage
    stage_input_names = list(input_names)
    # reverse the stage so that we start constructing from backward stages
    # that way, we can know which variables are needed by rest stages, and put it into "needed_vars"
    # in stage function, we will dynamically discard unsed variables to reduce transmission between stages
    for function_list in stages_list[::-1]:
        # for the last stage, output will be in same sequence with output_vars
        stage = stage_wrapper(function_list, needed_vars, stage_input_names)
        computational_stages.append(stage)
        needed_vars = set(needed_vars)
        for func in function_list:
            needed_vars = needed_vars.union(set(get_var_list(func)))
        # sort needed_varss so that it won't need to recompile because of output order changing
        needed_vars = sorted(needed_vars)
    # reverse computational_stages, because it's generated in reverse sequence
    return computational_stages[::-1]


def convolutional(filters_shape,
                  trainable,
                  use_gn,
                  name,
                  precision=tf.float16,
                  downsample=False,
                  activate=True,
                  norm=True,
                  weight_centering=False):
    scope_name = tf.get_variable_scope().name

    def conv_wrapper(input_data):
        with tf.variable_scope(scope_name):
            input_data = common.convolutional(input_data,
                                              filters_shape,
                                              trainable,
                                              use_gn,
                                              name,
                                              precision,
                                              downsample,
                                              activate,
                                              norm,
                                              weight_centering)
            return {"input_data": input_data}
    return conv_wrapper


def residual_block(input_channel, filter_num1, filter_num2, trainable, use_gn, name, precision):
    scope_name = tf.get_variable_scope().name

    def residual_layer1_wrapper(input_data):
        with tf.variable_scope(scope_name):
            with tf.variable_scope(name):
                short_cut = input_data
                input_data = common.convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                                  trainable=trainable, use_gn=use_gn, name='conv1', precision=precision)
                return {'short_cut': short_cut, 'input_data': input_data}

    def residual_layer2_wrapper(short_cut, input_data):
        with tf.variable_scope(scope_name):
            with tf.variable_scope(name):
                input_data = common.convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                                  trainable=trainable, use_gn=use_gn, name='conv2', precision=precision)
                input_data = input_data + short_cut
                # "short_cut" won't be used in the following functions
                # set short_cut to None means this value will be removed from stage function's output
                return {'short_cut': None, 'input_data': input_data}

    return [residual_layer1_wrapper, residual_layer2_wrapper]


def route(wrapper_func, route_name):

    def route_wrapper(short_cut, input_data):
        # all routes come after residual block
        # so the function input should be short_cut + input_data
        result = wrapper_func(short_cut, input_data)
        input_data = result['input_data']
        return {route_name: input_data, **result}

    return route_wrapper


def branch(wrapper_func, branch_name):

    def branch_wrapper(input_data):
        # branch wrapper is used before yolov3 predict head
        input_data = wrapper_func(input_data)['input_data']
        return {branch_name: input_data, 'input_data': input_data}

    return branch_wrapper
