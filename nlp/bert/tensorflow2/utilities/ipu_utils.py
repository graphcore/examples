# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import random

import numpy as np
import popdist.tensorflow
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.distributed.popdist_strategy import PopDistStrategy


def create_ipu_strategy(
    num_ipus_per_replica,
    num_replicas,
    distributed_training=False,
    fp_exceptions=False,
    enable_recomputation=True,
    enable_stochastic_rounding=True,
    min_remote_tensor_size=50000,
    max_cross_replica_sum_buffer_size=10 * 1024 * 1024,
    compile_only=False,
):
    """
    Creates an IPU config and returns an IPU strategy ready to run
    something on IPUs
    :param num_ipus_per_replica: Int representing the number of IPUs required per replica.
    :param num_replicas: Int representing the number of replicas required.
    :param distributed_training: Bool, indicates if script is launched with poprun tool for distributed training in
        possibly multiple PODs.
    :param fp_exceptions: Bool, if True floating point exceptions will be raised.
    :param enable_recomputation: Bool, if True recomputation will be enabled.
    :param enable_stochastic_rounding: Bool, if True, stochastic rounding
        will be enabled. This is recommended for training in fp16.
    :param min_remote_tensor_size: The minimum size (in bytes) a tensor
        must be in order to be considered for being stored in remote memory.
    :param max_cross_replica_sum_buffer_size: The maximum number of bytes
        that can be waiting before a cross replica sum op is scheduled.
        Represents an always-live vs not-always-live trade off. The
        default used here is effective for BERT.
    :return: An IPU strategy
    """
    ipu_config = ipu.config.IPUConfig()

    # Enable / disable floating point exceptions.
    ipu_config.floating_point_behaviour.inv = fp_exceptions
    ipu_config.floating_point_behaviour.div0 = fp_exceptions
    ipu_config.floating_point_behaviour.oflo = fp_exceptions
    ipu_config.floating_point_behaviour.nanoo = fp_exceptions

    # Enable / disable recomputation.
    ipu_config.allow_recompute = enable_recomputation

    # Enable / disable stochastic rounding.
    ipu_config.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.from_bool(
        enable_stochastic_rounding
    )

    # Enable / disable different optimisations.
    ipu_config.optimizations.minimum_remote_tensor_size = min_remote_tensor_size
    ipu_config.optimizations.merge_infeed_io_copies = True
    ipu_config.optimizations.maximum_cross_replica_sum_buffer_size = max_cross_replica_sum_buffer_size

    # Configure connection to device.
    if compile_only:
        ipu_config.device_connection.version = "ipu2"
        ipu_config.device_connection.type = ipu.config.DeviceConnectionType.PRE_COMPILE
    ipu_config.device_connection.enable_remote_buffers = True

    if distributed_training:
        if num_replicas != popdist.getNumTotalReplicas():
            logging.error(
                f"Replication factor given to poprun (=={popdist.getNumTotalReplicas()}) "
                f"does not match the config (=={num_replicas})."
            )
        logging.info(f"Total number of instances {popdist.getNumInstances()}")

        popdist.tensorflow.set_ipu_config(ipu_config, ipus_per_replica=num_ipus_per_replica, configure_device=True)
    else:
        ipu_config.auto_select_ipus = num_ipus_per_replica * num_replicas

    ipu_config.configure_ipu_system()

    strategy = PopDistStrategy() if distributed_training else ipu.ipu_strategy.IPUStrategy()
    return strategy


def get_poplar_options_per_pipeline_stage(
    num_ipus_per_replica, device_mapping, matmul_available_memory_proportion, matmul_partials_type
):
    if len(matmul_available_memory_proportion) != num_ipus_per_replica:
        raise ValueError(
            f"Available memory proportion must be set for each of the {num_ipus_per_replica} IPUs in the pipeline."
        )

    return [
        ipu.pipelining_ops.PipelineStageOptions(
            matmul_options={
                "availableMemoryProportion": str(matmul_available_memory_proportion[stage]),
                "partialsType": matmul_partials_type,
            }
        )
        for stage in device_mapping
    ]


def set_random_seeds(seed=42):
    ipu.utils.reset_ipu_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
