# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import random

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu


def create_ipu_strategy(num_ipus_per_replica,
                        num_replicas,
                        compile_only=False):
    """
    Creates an IPU config and returns an IPU strategy ready to run
    something on IPUs
    :param num_ipus_per_replica: Int representing the number of IPUs required per replica.
    :param num_replicas: Int representing the number of replicas required.
    :param compile_only: If only requiring compilation, this should be set
        to True.
    :return: An IPU strategy
    """
    ipu_config = ipu.config.IPUConfig()

    # Configure connection to device.
    ipu_config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
    if compile_only:
        ipu_config.device_connection.version = "ipu2"
        ipu_config.device_connection.type = ipu.config.DeviceConnectionType.PRE_COMPILE
    ipu_config.device_connection.enable_remote_buffers = True

    ipu_config.auto_select_ipus = num_ipus_per_replica * num_replicas

    ipu_config.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    return strategy


def set_random_seeds(seed=42):
    ipu.utils.reset_ipu_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
