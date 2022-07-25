# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import os

import popdist
import popdist.tensorflow
from tensorflow.python import ipu
from tensorflow.python.ipu.config import (DeviceConnectionType,
                                          SchedulingAlgorithm,
                                          StochasticRoundingBehaviour)


def configure_ipu(hparams) -> ipu.config.IPUConfig:
    cfg = ipu.config.IPUConfig()

    cfg.allow_recompute = hparams.recomputation

    if hparams.compile_only:
        cfg.device_connection.version = 'ipu2'
        cfg.device_connection.enable_remote_buffers = True
        # PRE_COMPILE allows for runing executables on graph without being online
        cfg.device_connection.type = DeviceConnectionType.PRE_COMPILE

        # Enforce using a exe cache path, defaulting if it doesnt exist
        tf_poplar_flags = os.environ.get('TF_POPLAR_FLAGS') or ''
        if '--executable_cache_path' not in tf_poplar_flags:
            logging.warning('Warning: --executable_cache_path not set. Defaulting to \'/tmp/tf2_cache\'.')
            tf_poplar_flags = f"{tf_poplar_flags} --executable_cache_path=/tmp/tf2_cache"
            os.environ["TF_POPLAR_FLAGS"] = tf_poplar_flags

    cfg.floating_point_behaviour.inv = hparams.fp_exceptions
    cfg.floating_point_behaviour.div0 = hparams.fp_exceptions
    cfg.floating_point_behaviour.oflo = hparams.fp_exceptions
    cfg.floating_point_behaviour.nanoo = hparams.nanoo
    cfg.floating_point_behaviour.esr = next(p for p in list(StochasticRoundingBehaviour)
                                            if hparams.stochastic_rounding == str(p).split(".")[-1])

    cfg.experimental.enable_prng_stability = True if hparams.seed else False
    cfg.norms.experimental.distributed_batch_norm_replica_group_size = hparams.dbn_replica_group_size
    cfg.norms.use_stable_statistics = hparams.stable_norm

    cfg.optimizations.merge_infeed_io_copies = True
    cfg.optimizations.minimum_remote_tensor_size = hparams.min_remote_tensor_size
    cfg.optimizations.maximum_reduce_many_buffer_size = hparams.max_reduce_many_buffer_size
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = hparams.max_cross_replica_buffer_size

    cfg.convolutions.poplar_options['gatherConvOutput'] = 'true' if hparams.gather_conv_output else 'false'
    cfg.convolutions.poplar_options['enableConvDithering'] = 'true' if hparams.conv_dithering else 'false'

    cfg.scheduling.algorithm = next(p for p in list(SchedulingAlgorithm)
                                    if hparams.scheduling_algorithm == str(p).split(".")[-1])

    if hparams.on_demand and not hparams.compile_only:
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND

    if hparams.half_partials:
        cfg.matmuls.poplar_options['partialsType'] = 'half'
        cfg.convolutions.poplar_options['partialsType'] = 'half'

    if len(hparams.available_memory_proportion) == 1:
        cfg.matmuls.poplar_options['availableMemoryProportion'] = str(hparams.available_memory_proportion[0] / 100)
        cfg.convolutions.poplar_options['availableMemoryProportion'] = str(hparams.available_memory_proportion[0] / 100)

    if hparams.seed is None and not hparams.distributed_training:
        cfg.compilation_poplar_options['target.deterministicWorkers'] = 'false'
    else:
        cfg.compilation_poplar_options['target.deterministicWorkers'] = 'portable'

    if hparams.internal_exchange_optimization_target is not None:
        cfg.compilation_poplar_options['opt.internalExchangeOptimisationTarget'] = hparams.internal_exchange_optimization_target

    if hparams.training:
        if hparams.distributed_training:
            popdist.tensorflow.set_ipu_config(cfg, ipus_per_replica=hparams.num_ipus_per_replica, configure_device=True)
        else:
            cfg.auto_select_ipus = hparams.num_ipus_per_replica * hparams.num_replicas

        cfg.configure_ipu_system()

    return cfg
