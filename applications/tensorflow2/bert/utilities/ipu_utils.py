# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import random

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu


def create_ipu_strategy(num_ipus,
                        fp_exceptions=False,
                        enable_recomputation=True,
                        min_remote_tensor_size=50000,
                        max_cross_replica_sum_buffer_size=10*1024*1024):
    """
    Creates an IPU config and returns an IPU strategy ready to run
    something on IPUs
    :param num_ipus: Int representing the number of IPUs required.
    :param fp_exceptions: Bool, if True floating point exceptions will
        be raised.
    :param enable_recomputation: Bool, if True recomputation will be
        enabled.
    :param min_remote_tensor_size: The minimum size (in bytes) a tensor
        must be in order to be considered for being stored in remote
        memory.
    :param max_cross_replica_sum_buffer_size: The maximum number of bytes
        that can be waiting before a cross replica sum op is scheduled.
        Represents an always-live vs not-always-live trade off. The
        default used here is effective for BERT.
    :return: An IPU strategy
    """
    ipu_config = ipu.config.IPUConfig()

    ipu_config.auto_select_ipus = num_ipus

    ipu_config.allow_recompute = enable_recomputation

    ipu_config.floating_point_behaviour.inv = fp_exceptions
    ipu_config.floating_point_behaviour.div0 = fp_exceptions
    ipu_config.floating_point_behaviour.oflo = fp_exceptions
    ipu_config.floating_point_behaviour.nanoo = fp_exceptions

    ipu_config.optimizations.minimum_remote_tensor_size = min_remote_tensor_size
    ipu_config.optimizations.merge_infeed_io_copies = True
    ipu_config.optimizations.maximum_cross_replica_sum_buffer_size = max_cross_replica_sum_buffer_size

    ipu_config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    ipu_config.device_connection.enable_remote_buffers = True

    ipu_config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    return strategy


def get_poplar_options_per_pipeline_stage(num_ipus_per_replica,
                                          device_mapping,
                                          matmul_available_memory_proportion,
                                          matmul_partials_type):
    if len(matmul_available_memory_proportion) != num_ipus_per_replica:
        raise ValueError(
            f"Available memory proportion must be set for each of the {num_ipus_per_replica} IPUs in the pipeline.")

    return [
        ipu.pipelining_ops.PipelineStageOptions(
            matmul_options={
                "availableMemoryProportion": str(matmul_available_memory_proportion[stage]),
                "partialsType": matmul_partials_type,
            }
        ) for stage in device_mapping]


def set_random_seeds(seed=42):
    ipu.utils.reset_ipu_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
