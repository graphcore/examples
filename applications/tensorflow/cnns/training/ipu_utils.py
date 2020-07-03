# Copyright 2019 Graphcore Ltd.
import os
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.utils import ExecutionProfileType


def get_config(prng=False,
               ipu_id=-1,
               shards=1,
               number_of_replicas=1,
               max_cross_replica_buffer_size=10*1024*1024,
               merge_infeed_io_copies=True,
               fp_exceptions=True,
               xla_recompute=False,
               seed=None,
               profile=None,
               availableMemoryProportion=None,
               stable_norm=False):
    """Builds ipu_options"""

    profile_exec_modes = {"NO_PROFILE": ExecutionProfileType.NO_PROFILE,
                          "TILE_PROFILE": ExecutionProfileType.TILE_PROFILE,
                          "DEVICE_PROFILE": ExecutionProfileType.DEVICE_PROFILE,
                          "IPU_PROFILE": ExecutionProfileType.IPU_PROFILE}

    config = utils.create_ipu_config(max_cross_replica_sum_buffer_size=max_cross_replica_buffer_size,
                                     merge_infeed_io_copies=merge_infeed_io_copies,
                                     always_rearrange_copies_on_the_host=False,
                                     profiling=profile is not None,
                                     profile_execution=profile_exec_modes[profile] if profile else None)

    if "GCL_REAL_COLLECTIVES" in os.environ:
        config = utils.set_gcl_options(config, num_io_tiles=128, gcl_options={"useGclCollectives": "true", })

    if ipu_id == -1:
        config = utils.auto_select_ipus(config, number_of_replicas*shards)
    else:
        config = utils.select_ipus(config, [ipu_id])
    config = utils.set_compilation_options(config, {
        "device.clearAtomicFlagAfterExchange": "false",
        "prng.enable": "true" if prng else "false",
        "target.deterministicWorkers": "false" if seed is None else "true",
    })

    if availableMemoryProportion is not None:
        config = utils.set_convolution_options(config, {
            "availableMemoryProportion": str(availableMemoryProportion)
        })

    if stable_norm:
        config = utils.set_norm_options(config, use_stable_statistics=True)

    if xla_recompute:
        utils.set_recomputation_options(config, allow_recompute=True)

    config = utils.set_floating_point_behaviour_options(config, inv=fp_exceptions, div0=fp_exceptions,
                                                        oflo=fp_exceptions, esr=prng, nanoo=True)

    return config
