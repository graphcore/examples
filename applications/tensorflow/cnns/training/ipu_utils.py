# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

import os
import subprocess
import re
from tensorflow.python.ipu.config import (
    IPUConfig,
    SchedulingAlgorithm,
    DeviceConnectionType
)
from tensorflow.python.ipu import utils



def get_ipu_arch():
    try:
        cmd = ['gc-info', '-d', '0', '--ipu-arch']
        ret = subprocess.check_output(cmd).decode('utf-8')

        pattern = re.compile('ipu(\d)')
        match = pattern.match(ret)
        arch = int(match.group(1))
    except:
        arch = 2
    return arch


def get_config(prng=False,
               ipu_id=-1,
               shards=1,
               number_of_replicas=1,
               max_cross_replica_buffer_size=50*1024*1024,
               merge_infeed_io_copies=True,
               fp_exceptions=True,
               half_partials=False,
               conv_dithering=False,
               conv_output=False,
               enable_recomputation=False,
               seed=None,
               availableMemoryProportion=None,
               stable_norm=False,
               internalExchangeOptimisationTarget=None,
               num_io_tiles=0,
               number_of_distributed_batch_norm_replicas=1,
               min_remote_tensor_size=128,
               compile_only=False,
               nanoo=True,
               scheduling_algorithm=SchedulingAlgorithm.CHOOSE_BEST,
               max_reduce_many_buffer_size=0
               ):
    """Builds ipu_options"""
    config = IPUConfig()

    config.optimizations.merge_infeed_io_copies = merge_infeed_io_copies
    if scheduling_algorithm == SchedulingAlgorithm.CHOOSE_BEST:
        if get_ipu_arch() == 2:
            scheduling_algorithm = SchedulingAlgorithm.SHORTEST_PATH
        else:
            # work around to avoid OOM on MK1
            scheduling_algorithm = SchedulingAlgorithm.CHOOSE_BEST
    config.scheduling.algorithm = scheduling_algorithm
    config.experimental.always_rearrange_copies_on_the_host = False
    config.optimizations.minimum_remote_tensor_size = min_remote_tensor_size
    config.optimizations.maximum_cross_replica_sum_buffer_size = (
        max_cross_replica_buffer_size)
    config.optimizations.maximum_reduce_many_buffer_size = (
        max_reduce_many_buffer_size)

    if ipu_id == -1:
        config.auto_select_ipus = number_of_replicas * shards
    else:
        config.select_ipus = [ipu_id]
    config.compilation_poplar_options = {
        'target.deterministicWorkers': 'false' if seed is None else 'portable'}

    if internalExchangeOptimisationTarget is not None:
        config.compilation_poplar_options['opt.internalExchangeOptimisationTarget'] = internalExchangeOptimisationTarget

    if num_io_tiles != 0:
        config.io_tiles.place_ops_on_io_tiles = True
        config.io_tiles.num_io_tiles = num_io_tiles

    config.convolutions.poplar_options = {}

    if availableMemoryProportion is not None:
        config.convolutions.poplar_options['availableMemoryProportion'] = str(availableMemoryProportion)

    if half_partials:
        config.convolutions.poplar_options['partialsType'] = 'half'
        config.matmuls.poplar_options['partialsType'] = 'half'
    if conv_dithering:
        config.convolutions.poplar_options['enableConvDithering'] = 'true'
    if conv_output:
        config.convolutions.poplar_options['gatherConvOutput'] = 'true'

    if stable_norm:
        config.norms.use_stable_statistics = True

    if enable_recomputation:
        config.allow_recompute = True

    if compile_only:
        config.device_connection.version = 'ipu2'
        config.device_connection.enable_remote_buffers = True
        # PRE_COMPILE allows for runing execuatables on graph without being online
        config.device_connection.type = DeviceConnectionType.PRE_COMPILE

        # Enforce using a exe cache path, defaulting if it doesnt exist
        tf_poplar_flags = os.environ.get("TF_POPLAR_FLAGS") or ''

        if '--executable_cache_path' not in tf_poplar_flags:
            print("Warning: --executable_cache_path not set. " +
                  "Defaulting to '/tmp/tf_cache'.")

            tf_poplar_flags = f"{tf_poplar_flags} --executable_cache_path=/tmp/tf_cache"
            os.environ["TF_POPLAR_FLAGS"] = tf_poplar_flags

    config.floating_point_behaviour.inv = fp_exceptions
    config.floating_point_behaviour.div0 = fp_exceptions
    config.floating_point_behaviour.oflo = fp_exceptions
    config.floating_point_behaviour.esr = prng
    config.floating_point_behaviour.nanoo = nanoo

    config.norms.experimental.distributed_batch_norm_replica_group_size = (
        number_of_distributed_batch_norm_replicas)

    return config
