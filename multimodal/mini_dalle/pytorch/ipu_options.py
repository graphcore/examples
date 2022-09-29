# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import poptorch
import popart
import popdist
import torch


def get_options(config):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''

    # Numpy options
    np.random.seed(config.random_seed)

    # Poptorch options
    if config.use_popdist:
        # Use popdist.poptorch options if running in distributed mode
        opts = popdist.poptorch.Options(ipus_per_replica=config.ipus_per_replica)
    else:
        opts = poptorch.Options()
        # Set the replication factor
        opts.replicationFactor(config.replication_factor)

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.device_iterations)

    # Set gradient accumulation factors
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

    # Enable automatic loss scaling
    opts.Training.setAutomaticLossScaling(config.enable_als)

    # Return all results from IPU to host
    opts.outputMode(poptorch.OutputMode.Final)

    # Fix the random seeds
    opts.randomSeed(config.random_seed)

    # Enable Replicated Tensor Sharding (RTS) of optimizer state
    #  with optimizer state residing either on-chip or in DRAM
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        # Optimizer state lives on-chip
        .useOnChipStorage(not config.optimizer_state_offchip)
        # Shard optimizer state between replicas with zero-redundancy
        .useReplicatedTensorSharding(config.enable_rts))

    # Disable warnings of unexpected optimizer attributes
    opts.relaxOptimizerAttributesChecks()

    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    # Compile offline (no IPUs required)
    if config.compile_only:
        opts.useOfflineIpuTarget()

    # Set available Transient Memory For matmuls and convolutions operations
    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)

    # Enable stochastic rounding (recommended for training with FP16)
    opts.Precision.enableStochasticRounding(config.stochastic_rounding)

    # Enable caching the compiled executable to disk
    if config.executable_cache_dir:
        opts.enableExecutableCaching(config.executable_cache_dir)

    # Half precision partials for matmuls and convolutions
    if config.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)
    else:
        opts.Precision.setPartialsType(torch.float32)

    # PopART performance options #
    # Only stream needed tensors back to host
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("outlineThreshold", 10.0)

    # Options for profiling with Popvision
    engine_options = {
        "opt.useAutoloader": "true",
        "target.syncReplicasIndependently": "true",
    }
    if config.profile_dir:
        engine_options = {
            **engine_options,
            **{
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": config.profile_dir,
                "profiler.format": "v3",
                "autoReport.all": "true",
            }
        }
    opts._Popart.set("engineOptions", engine_options)

    # Enable synthetic random data generated on device (so with no I/O)
    if config.synthetic_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))

    return opts
