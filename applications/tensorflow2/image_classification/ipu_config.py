# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import popdist
import popdist.tensorflow
from tensorflow.python import ipu
from tensorflow.python.ipu.config import StochasticRoundingBehaviour


AVAILABLE_SR_OPTIONS = {
    'ON': StochasticRoundingBehaviour.ON,
    'OFF': StochasticRoundingBehaviour.OFF,
    'RI': StochasticRoundingBehaviour.REPLICA_IDENTICAL_ONLY
}


def configure_ipu(hparams) -> ipu.config.IPUConfig:
    cfg = ipu.config.IPUConfig()

    cfg.allow_recompute = hparams.recomputation
    cfg.optimizations.merge_infeed_io_copies = True
    cfg.norms.use_stable_statistics = hparams.stable_norm
    cfg.floating_point_behaviour.inv = hparams.fp_exceptions
    cfg.floating_point_behaviour.div0 = hparams.fp_exceptions
    cfg.floating_point_behaviour.oflo = hparams.fp_exceptions
    cfg.experimental.enable_prng_stability = True if hparams.seed else False
    cfg.optimizations.maximum_reduce_many_buffer_size = hparams.max_reduce_many_buffer_size
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = hparams.max_cross_replica_buffer_size
    cfg.norms.experimental.distributed_batch_norm_replica_group_size = hparams.dbn_replica_group_size
    cfg.floating_point_behaviour.esr = AVAILABLE_SR_OPTIONS[hparams.stochastic_rounding]

    if hparams.gather_conv_output:
        cfg.convolutions.poplar_options['gatherConvOutput'] = 'true'

    if hparams.on_demand:
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
