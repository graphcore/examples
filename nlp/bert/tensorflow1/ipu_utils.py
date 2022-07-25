# coding=utf-8
# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import tensorflow as tf
from tensorflow.python.ipu.config import IPUConfig, DeviceConnectionType, SchedulingAlgorithm, MergeRemoteBuffersBehaviour, StochasticRoundingBehaviour
from collections import OrderedDict
from tensorflow.python.ipu import horovod as hvd


def barrier():
    with tf.Graph().as_default(), tf.Session():
        _ = hvd.allreduce(tf.constant(42.0, name="Broadcast", dtype = tf.float16), op=hvd.Sum).eval()


def next_power_of_two(x):
    return 2**int(math.ceil(math.log2(x)))


def get_config(fp_exceptions,
               enable_recomputation,
               disable_graph_outlining,
               num_required_ipus,
               enable_stochastic_rounding,
               max_cross_replica_sum_buffer_size,
               max_reduce_scatter_buffer_size,
               scheduler_selection,
               compile_only,
               ipu_id,
               available_memory_proportion=None,
               partials_type="half",
               minimum_remote_tensor_size=128):

    # Builds ipu_options
    cfg = IPUConfig()

    if ipu_id:
        cfg.select_ipus = [ipu_id]
    else:
        cfg.auto_select_ipus = num_required_ipus

    cfg.allow_recompute = enable_recomputation
    cfg.scheduling.algorithm = SchedulingAlgorithm[scheduler_selection]
    cfg.norms.use_stable_statistics = True
    cfg.matmuls.clear_pass_type = True

    if compile_only:
        cfg.device_connection.type = tf.python.ipu.utils.DeviceConnectionType.PRE_COMPILE
        cfg.device_connection.version = "ipu2"
        cfg.device_connection.enable_remote_buffers = True

    # Floating-point exceptions
    cfg.floating_point_behaviour.inv = fp_exceptions
    cfg.floating_point_behaviour.div0 = fp_exceptions
    cfg.floating_point_behaviour.oflo = fp_exceptions
    cfg.floating_point_behaviour.nanoo = fp_exceptions

    # Stochastic rounding
    if enable_stochastic_rounding:
        cfg.floating_point_behaviour.esr = StochasticRoundingBehaviour.ON
    else:
        cfg.floating_point_behaviour.esr = StochasticRoundingBehaviour.OFF
    cfg.optimizations.merge_remote_buffers = MergeRemoteBuffersBehaviour.MERGE
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = max_cross_replica_sum_buffer_size
    cfg.optimizations.maximum_reduce_scatter_buffer_size = max_reduce_scatter_buffer_size
    cfg.optimizations.merge_infeed_io_copies = True
    cfg.optimizations.enable_graph_outlining = not disable_graph_outlining
    cfg.optimizations.minimum_remote_tensor_size = minimum_remote_tensor_size

    if available_memory_proportion is not None:
        cfg.convolutions.poplar_options = {
            "availableMemoryProportion": str(available_memory_proportion),
            "partialsType": partials_type
        }
        cfg.matmuls.poplar_options = {
            "availableMemoryProportion": str(available_memory_proportion),
            "partialsType": partials_type
        }

    return cfg


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
            layer's output must be a dictionary so that stage_function will know which param is needed by rest layers
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
            # assume outputs to be a dictionary
            assert isinstance(outputs, dict)
            result.update(outputs)
        # only return needed vlaues
        result = OrderedDict([(key, result[key])
                              for key in needed_vars if key in result.keys()])
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
