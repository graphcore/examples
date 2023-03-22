# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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
    matmul_available_memory_proportion,
    matmul_partials_type,
    fp_exceptions,
    distributed_training=False,
    enable_recomputation=True,
    num_io_tiles=0,
    compile_only=False,
):
    """
    Creates an IPU config and returns an IPU strategy ready to run
    something on IPUs
    :param num_ipus_per_replica: Int representing the number of IPUs
        required per replica.
    :param num_replicas: Int representing the number of replicas required.
    :param compile_only: If only requiring compilation, this should be set
        to True.
    :param matmul_available_memory_proportion: A float which determines
        the matmul available memory proportion for the model. If using
        pipelining, this will be overridden by the setting per pipeline
        stage.
    :param matmul_partials_type: A string which determines type of the
        intermediate calculations of the matmuls. If using pipelining,
        this will be overridden by the setting per pipeline stage.
    :param fp_exceptions: A boolean to turn floating point exceptions on or off.
    :param distributed_training: Bool, indicates if script is launched with poprun
        tool for distributed training in possibly multiple PODs.
    :param enable_recomputation: A flag to turn on recomputation.
    :param num_io_tiles: Designate this amount of tiles on an IPU exclusively
        for IO. This can overlap the data transfer time and the compute.
    :param compile_only: A boolean to turn on whether compilation should
        happen without running anything after. Set env var
        TF_POPLAR_FLAGS=--executable_cache_path=/path/to/storage before
        using this.
    :return: An IPU strategy
    """
    ipu_config = ipu.config.IPUConfig()

    # Enable / disable recomputation.
    ipu_config.allow_recompute = enable_recomputation

    # Enable / disable floating point exceptions.
    ipu_config.floating_point_behaviour.inv = fp_exceptions
    ipu_config.floating_point_behaviour.div0 = fp_exceptions
    ipu_config.floating_point_behaviour.oflo = fp_exceptions
    ipu_config.floating_point_behaviour.nanoo = fp_exceptions

    # Configure connection to device.
    if compile_only:
        ipu_config.device_connection.version = "ipu2"
        ipu_config.device_connection.type = ipu.config.DeviceConnectionType.PRE_COMPILE
    ipu_config.device_connection.enable_remote_buffers = True

    # Set the matmul poplar options. If pipelining is used these
    # will be overridden per pipeline stage during pipeline assignment.
    ipu_config.matmuls.poplar_options = get_matmul_options(matmul_available_memory_proportion, matmul_partials_type)

    if num_io_tiles:
        ipu_config.io_tiles.num_io_tiles = num_io_tiles
        ipu_config.io_tiles.place_ops_on_io_tiles = True

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


def get_matmul_options(matmul_available_memory_proportion, matmul_partials_type):
    return {"availableMemoryProportion": str(matmul_available_memory_proportion), "partialsType": matmul_partials_type}


def set_random_seeds(seed=42):
    ipu.utils.reset_ipu_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
