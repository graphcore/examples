# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
A portable API for IPU or non-IPU code.

Note: Assumes you'll use the IPU whenever gc-tensorflow is installed.
"""

import tensorflow as tf

try:
    from tensorflow.python import ipu

    IS_IPU = True
except ImportError:

    IS_IPU = False

from tensorflow import gather

if IS_IPU:
    from ipu_tensorflow_addons.keras.layers import Dropout, LayerNormalization
    from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer
    from tensorflow.python.ipu import nn_ops

    outlined_function = ipu.outlined_function

    from static_ops.static_ops import grouped_gather, grouped_scatter_max, grouped_scatter_sum

    gelu = nn_ops.gelu
    from tensorflow.keras.ipu import PipelineStage as PipelineStageInner

    def PipelineStage(stage, num_ipus):
        if num_ipus > 1:
            return PipelineStageInner(stage)
        return DummyScope()

else:
    from tensorflow.keras.activations import gelu
    from tensorflow.keras.layers import Dropout, LayerNormalization
    from tensorflow.keras.optimizers import Adam

    outlined_function = identity

    def PipelineStage(*_):
        return DummyScope()

    from model.mpnn.layers import _scatter

    # uses list comprehension over the first dimension
    # note: these are implemented for cross-compatibility and are slow
    def grouped_scatter(data: tf.Tensor, indices: tf.Tensor, table_size: int) -> tf.Tensor:
        return _scatter(data, indices, table_size, gather_scatter_method="debug")

    def grouped_gather(params: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
        return gather(params, indices, batch_dims=1)


class DummyScope:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyStrategy:
    def __init__(self):
        self.scope = DummyScope


def configure_and_get_strategy(num_replicas, num_ipus_per_replica, cfg, stochastic_rounding=True):
    if IS_IPU:
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = num_replicas * num_ipus_per_replica
        config.matmuls.poplar_options["partialsType"] = "half"
        available_memory_proportion = cfg.ipu_opts.available_memory_proportion[0]
        config.matmuls.poplar_options["availableMemoryProportion"] = str(available_memory_proportion)
        config.slices.poplar_options["availableMemoryProportion"] = str(available_memory_proportion)
        # balanced, memory or cycles
        config.compilation_poplar_options["opt.internalExchangeOptimisationTarget"] = cfg.ipu_opts.optimization_target
        config.scheduling.algorithm = vars(ipu.config.SchedulingAlgorithm)[cfg.ipu_opts.scheduling_algorithm]
        config.optimizations.maximum_cross_replica_sum_buffer_size = cfg.ipu_opts.maximum_cross_replica_sum_buffer_size
        config.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
        config.device_connection.version = "ipu2"
        config.device_connection.enable_remote_buffers = True

        if stochastic_rounding:
            config.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.ON
        else:
            # Turn stochastic rounidng off during inference
            config.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.OFF

        if cfg.ipu_opts.fp_exceptions:
            config.floating_point_behaviour.inv = True
            config.floating_point_behaviour.div0 = True
            config.floating_point_behaviour.oflo = True
            config.floating_point_behaviour.nanoo = True
        elif cfg.ipu_opts.nanoo:
            config.floating_point_behaviour.nanoo = True

        config.allow_recompute = cfg.ipu_opts.recompute

        ipu.utils.configure_ipu_system(config)
        strategy = ipu.ipu_strategy.IPUStrategy()
    else:
        strategy = DummyStrategy()

    return strategy


def call_outlined_function(f, *args, **kwargs):
    """Wraps ipu.outlined_function to handle positional, keyword and non-Tensor arguments."""
    tensor_keys = [k for k, v in kwargs.items() if isinstance(v, tf.Tensor)]

    @outlined_function
    def wrapper(*_args):
        pos_args = _args[: len(args)]
        kw_args = _args[len(args) :]
        fnargs = kwargs.copy()
        fnargs.update(dict(zip(tensor_keys, kw_args)))
        return f(*pos_args, **fnargs)

    all_args = args + tuple(kwargs[k] for k in tensor_keys)
    return wrapper(*all_args)
