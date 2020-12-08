# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
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


import itertools
import math

import tensorflow as tf
from tensorflow.python.ipu import outlined_function as ipu_function
from tensorflow.python.ipu import utils


def next_power_of_two(x):
    return 2**int(math.ceil(math.log2(x)))


def ladder_numbering_iterator():
    """Generate an IPU ladder-style numbering [0, 1, 3, 2, 4, 5, 7, 6...].
    returns -- generator(int)
    """
    return (x ^ ((x & 0x02) >> 1) for x in itertools.count())


def get_ipu_config(fp_exceptions=True,
                   stochastic_rounding=True,
                   xla_recompute=False,
                   available_memory_proportion=None,
                   disable_graph_outlining=False,
                   num_ipus_required=0,
                   max_cross_replica_sum_buffer_size=0,
                   scheduler_selection='',
                   compile_only=False,
                   partials_type="half"):
    """Builds ipu_options"""
    config = utils.create_ipu_config(
        max_report_size=3001819596000,
        merge_infeed_io_copies=True,
        always_rearrange_copies_on_the_host=False,
        selection_order=utils.SelectionOrder.AUTO,
        disable_graph_outlining=disable_graph_outlining,
        max_cross_replica_sum_buffer_size=max_cross_replica_sum_buffer_size,
        scheduler_selection=scheduler_selection
    )

    config = utils.auto_select_ipus(config, num_ipus_required)

    config = utils.set_matmul_options(config, clear_pass_type=True)

    if available_memory_proportion is not None:
        config = utils.set_convolution_options(config, {
            "availableMemoryProportion": str(available_memory_proportion),
            "partialsType": partials_type
        })
        config = utils.set_matmul_options(config, {
            "availableMemoryProportion": str(available_memory_proportion),
            "partialsType": partials_type
        })

    config = utils.set_norm_options(config, use_stable_statistics=True)

    config = utils.set_recomputation_options(config, allow_recompute=xla_recompute)

    if compile_only:
        config = utils.set_ipu_connection_type(config, utils.DeviceConnectionType.NEVER, ipu_version=2, enable_remote_buffers=True)

    config = utils.set_floating_point_behaviour_options(config, inv=fp_exceptions, div0=fp_exceptions,
                                                        oflo=fp_exceptions, esr=stochastic_rounding,
                                                        nanoo=fp_exceptions)
    return config


def function_decorator(use_ipu_function, func=None):
    """Wrapper which uses @ipu.function when enabled, and not otherwise"""
    def decorated(inner_func):
        if use_ipu_function:
            return ipu_function(inner_func)
        return inner_func

    if func is not None:
        return decorated(func)
    return decorated
