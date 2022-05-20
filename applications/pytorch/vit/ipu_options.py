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

import os
import ctypes
import numpy as np
import poptorch
import popart
import torch

import popdist.poptorch


def create_options(opts):
    if opts.use_popdist:
        model_opts = popdist.poptorch.Options()
    else:
        model_opts = poptorch.Options()

    return model_opts


def get_options(config):
    """
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    """

    # Poptorch options
    opts = create_options(config)
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.batches_per_step)

    # Set replication and gradient accumulation factors
    if not config.use_popdist:
        opts.replicationFactor(config.replication_factor)
    opts.Training.gradientAccumulation(config.gradient_accumulation)

    if config.reduction_type == "sum":
        opts.Training.accumulationAndReplicationReductionType(
            poptorch.ReductionType.Sum)
    elif config.reduction_type == "mean":
        opts.Training.accumulationAndReplicationReductionType(
            poptorch.ReductionType.Mean)
    else:
        raise ValueError(
            "Expected reduction type to be 'sum' or 'mean', but got %s" % config.reduction_type)

    # Enable automatic loss scaling
    # Note that this is an experimental feature. Note also that it expects
    # accumulationAndReplicationReductionType to be set to Mean as above,
    # and for accumulation by the optimizer to be done in half precision
    # using accum_type=torch.float16 during optimizer instatiation.
    if config.auto_loss_scaling is True:
        opts.Training.setAutomaticLossScaling(True)

    # Return all results from IPU to host
    opts.outputMode(poptorch.OutputMode.All)

    # Fix the random seeds
    np.random.seed(config.random_seed)
    opts.randomSeed(config.random_seed)

    # Enable Replicated Tensor Sharding (RTS) of optimizer state
    #  with optimizer state residing either on-chip or in DRAM
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        # Optimizer state lives on-chip
        .useOnChipStorage(not config.optimizer_state_offchip)
        # Shard optimizer state between replicas with zero-redundancy
        .useReplicatedTensorSharding(config.enable_rts))

    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    # Set available Transient Memory For matmuls and convolutions operations
    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)

    if config.synthetic_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))

    # Enable stochastic rounding (recommended for training with FP16)
    opts.Precision.enableStochasticRounding(config.stochastic_rounding)

    # Enable caching the compiled executable to disk
    if config.executable_cache_dir:
        opts.enableExecutableCaching(config.executable_cache_dir)

    # Half precision partials for matmuls and convolutions
    if config.half_partials:
        opts.Precision.setPartialsType(torch.half)

    # PopART performance options
    # Only stream needed tensors back to host
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))

    if config.prefetch_depth > 1:
        # How many batches to prefetch onto the IPU
        opts._Popart.set("defaultPrefetchBufferingDepth",
                         config.prefetch_depth)

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

    # Parallelize optimizer step update across IPUs
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set(
        "accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])

    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(
        popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)

    return opts
