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

import torch
import popart
import poptorch
import popdist
import popdist.poptorch
import numpy as np

from tools import logger


def get_options(config):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''

    # Numpy options
    np.random.seed(config.seed)

    # Load custom ops
    if config.custom_ops is True:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
        if os.path.exists(CUSTOM_OP_PATH):
            ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
        else:
            logger(
                "Could not find custom_ops.so. Execute `make` before running this script.")
            exit()

    # Poptorch options
    if config.use_popdist:
        # Use popdist.poptorch options if running in distributed mode
        opts = popdist.poptorch.Options(
            ipus_per_replica=config.ipus_per_replica)
    else:
        opts = poptorch.Options()
        # Set the replication factor
        opts.replicationFactor(config.replication_factor)

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.device_iterations)
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(
        poptorch.ReductionType.Mean)

    # Enable automatic loss scaling
    # Note that this is an experimental feature. Note also that it expects
    # accumulationAndReplicationReductionType to be set to Mean as above,
    # and for accumulation by the optimizer to be done in half precision
    # using accum_type=torch.float16 during optimizer instatiation.
    if config.auto_loss_scaling is True:
        opts.Training.setAutomaticLossScaling(True)

    opts.outputMode(poptorch.OutputMode.Sum)
    opts.TensorLocations.setOptimizerLocation(poptorch.TensorLocationSettings()
                                              .useOnChipStorage(not config.optimizer_state_offchip)
                                              .useReplicatedTensorSharding(config.replicated_tensor_sharding))
    opts.randomSeed(config.seed)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    if config.compile_only:
        opts.useOfflineIpuTarget()

    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)
    if config.executable_cache_dir:
        opts.enableExecutableCaching(config.executable_cache_dir)

    # Precision options
    opts.Precision.enableStochasticRounding(True)
    if config.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)

    # PopART options
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("outlineThreshold", 10.0)
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set(
        "accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(
        popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("decomposeGradSum", True)
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns(
        {"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True})

    opts._Popart.set("saveInitializersToFile", "weights.bin")

    return opts
