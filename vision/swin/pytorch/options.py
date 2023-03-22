# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import popart
import poptorch
import popdist.poptorch


def get_options(config):
    if popdist.isPopdistEnvSet():
        opts = popdist.poptorch.Options()
    else:
        opts = poptorch.Options()
        opts.replicationFactor(config.IPU.NUM_LOCALREPLICA)
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.IPU.DEVICE_ITERATIONS)
    opts.Training.gradientAccumulation(config.IPU.GRADIENT_ACCUMULATION_STEPS)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    opts._Popart.set("defaultBufferingDepth", 3)
    if config.IPU.NUM_LOCALREPLICA > 1 and config.TRAIN.OPTIMIZER.RTS:
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings()
            .useReplicatedTensorSharding(True)
            .minElementsForReplicatedTensorSharding(1024)
        )
    opts.randomSeed(0)
    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))
    ipu_list = [0] if config.IPU.LAYERS_PER_IPU is None else config.IPU.LAYERS_PER_IPU
    mem_prop = {f"IPU{i}": config.IPU.AMP for i, _ in enumerate(ipu_list)}

    opts.setAvailableMemoryProportion(mem_prop)
    opts.enableExecutableCaching("./cachedir")
    opts.Precision.enableStochasticRounding(True)
    if config.PRECISION[1] == "half":
        opts.Precision.setPartialsType(torch.half)
        opts.Training.setMeanAccumulationAndReplicationReductionStrategy(poptorch.MeanReductionStrategy.Running)
    else:
        opts.Precision.setPartialsType(torch.float32)

    if config.TRAIN.OPTIMIZER.RTS:
        opts._Popart.set("replicatedCollectivesSettings.prepareScheduleForMergingCollectives", True)
        opts._Popart.set("replicatedCollectivesSettings.mergeAllReduceCollectives", True)
    opts._Popart.set(
        "accumulateOuterFragmentSettings.schedule", int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized)
    )

    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns({"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True})

    return opts
