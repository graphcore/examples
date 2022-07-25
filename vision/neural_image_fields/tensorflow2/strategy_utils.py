# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf


# This strategy is used to minimise code
# differences in the GPU/CPU code path:
class NoStrategy():
    def __init__(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return

    def scope(self):
        return self


def get_train_strategy(no_ipu, replicas, fp16, nosre):
    # If not running on IPU we return a dummy strategy
    # so that external code is identical:
    if no_ipu:
        return NoStrategy()

    # Otherwise we configure the IPU strategy:
    from tensorflow.python import ipu
    cfg = ipu.config.IPUConfig()
    if fp16:
        # Half partials improve performance and do not usually
        # impact model quality:
        cfg.convolutions.poplar_options['partialsType'] = 'half'
        cfg.matmuls.poplar_options['partialsType'] = 'half'
    cfg.auto_select_ipus = replicas
    if nosre:
        cfg.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.OFF
    else:
        cfg.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.ON

    cfg.compilation_poplar_options = {"target.deterministicWorkers": "portable"}
    cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
    cfg.device_connection.enable_remote_buffers = True
    cfg.optimizations.math.fast = True
    cfg.configure_ipu_system()
    ipu.utils.reset_ipu_seed(101)
    return ipu.ipu_strategy.IPUStrategy()


def get_predict_strategy(no_ipu):
    if no_ipu:
        return NoStrategy()

    # Configure the IPU system:
    from tensorflow.python import ipu
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.compilation_poplar_options = {"target.deterministicWorkers": "portable"}
    cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
    cfg.device_connection.enable_remote_buffers = True
    cfg.optimizations.math.fast = True
    cfg.configure_ipu_system()
    return ipu.ipu_strategy.IPUStrategy()
