# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import os

import popdist
import popdist.tensorflow
from tensorflow.python import ipu
from tensorflow.python.ipu.config import DeviceConnectionType, SchedulingAlgorithm, StochasticRoundingBehaviour


def select_ipus(
    distributed_training: bool, num_replicas: int, num_ipus_per_replica: int, cfg=None
) -> ipu.config.IPUConfig:

    cfg = cfg or ipu.config.IPUConfig()

    if distributed_training:
        popdist.tensorflow.set_ipu_config(config=cfg, ipus_per_replica=num_ipus_per_replica, configure_device=True)
    else:
        cfg.auto_select_ipus = num_replicas * num_ipus_per_replica

    cfg.configure_ipu_system()

    return cfg


def configure_ipu(hparams, cfg=None, configure_ipu_system=True) -> ipu.config.IPUConfig:
    cfg = cfg or ipu.config.IPUConfig()

    cfg.allow_recompute = hparams.recomputation

    if hparams.compile_only:
        cfg.device_connection.version = "ipu2"
        cfg.device_connection.enable_remote_buffers = True
        # PRE_COMPILE allows for running executables on graph without being online
        cfg.device_connection.type = DeviceConnectionType.PRE_COMPILE

        # Enforce using a exe cache path, defaulting if it doesn't exist
        tf_poplar_flags = os.environ.get("TF_POPLAR_FLAGS") or ""
        if "--executable_cache_path" not in tf_poplar_flags:
            logging.warning("Warning: --executable_cache_path not set. Defaulting to '/tmp/tf2_cache'.")
            tf_poplar_flags = f"{tf_poplar_flags} --executable_cache_path=/tmp/tf2_cache"
            os.environ["TF_POPLAR_FLAGS"] = tf_poplar_flags

    cfg.floating_point_behaviour.inv = hparams.fp_exceptions
    cfg.floating_point_behaviour.div0 = hparams.fp_exceptions
    cfg.floating_point_behaviour.oflo = hparams.fp_exceptions
    cfg.floating_point_behaviour.nanoo = hparams.nanoo
    cfg.floating_point_behaviour.esr = next(
        p for p in list(StochasticRoundingBehaviour) if hparams.stochastic_rounding == str(p).split(".")[-1]
    )

    cfg.experimental.enable_prng_stability = True if hparams.seed else False
    cfg.norms.experimental.distributed_batch_norm_replica_group_size = hparams.dbn_replica_group_size
    cfg.norms.use_stable_statistics = hparams.stable_norm

    cfg.optimizations.merge_infeed_io_copies = True
    cfg.optimizations.minimum_remote_tensor_size = hparams.min_remote_tensor_size
    cfg.optimizations.maximum_reduce_many_buffer_size = hparams.max_reduce_many_buffer_size
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = hparams.max_cross_replica_buffer_size

    cfg.convolutions.poplar_options["gatherConvOutput"] = "true" if hparams.gather_conv_output else "false"
    cfg.convolutions.poplar_options["enableConvDithering"] = "true" if hparams.conv_dithering else "false"

    cfg.scheduling.algorithm = next(
        p for p in list(SchedulingAlgorithm) if hparams.scheduling_algorithm == str(p).split(".")[-1]
    )

    if hparams.on_demand and not hparams.compile_only:
        cfg.device_connection.enable_remote_buffers = True

    if hparams.half_partials:
        cfg.matmuls.poplar_options["partialsType"] = "half"
        cfg.convolutions.poplar_options["partialsType"] = "half"

    if len(hparams.available_memory_proportion) == 1:
        cfg.matmuls.poplar_options["availableMemoryProportion"] = str(hparams.available_memory_proportion[0] / 100)
        cfg.convolutions.poplar_options["availableMemoryProportion"] = str(hparams.available_memory_proportion[0] / 100)

    if hparams.seed is None and not hparams.distributed_training:
        cfg.compilation_poplar_options["target.deterministicWorkers"] = "false"
    else:
        cfg.compilation_poplar_options["target.deterministicWorkers"] = "portable"

    if hparams.internal_exchange_optimization_target is not None:
        cfg.compilation_poplar_options[
            "opt.internalExchangeOptimisationTarget"
        ] = hparams.internal_exchange_optimization_target

    assert hparams.synthetic_data in {
        "host",
        "ipu",
        None,
    }, f"Synthetic data option '{hparams.synthetic_data}' not recognized"
    if hparams.synthetic_data == "ipu":
        logging.info(f"Activating synthetic data on the ipu.")
        tf_poplar_flags = " --use_synthetic_data --synthetic_data_initializer=random"
        if "TF_POPLAR_FLAGS" in os.environ:
            os.environ["TF_POPLAR_FLAGS"] += tf_poplar_flags
        else:
            os.environ["TF_POPLAR_FLAGS"] = tf_poplar_flags

    if configure_ipu_system:
        cfg.configure_ipu_system()

    return cfg


def reconfigure_for_validation(cfg, configure_ipu_system=True):
    cfg.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.from_bool(False)
    if configure_ipu_system:
        cfg.configure_ipu_system()
    return cfg
