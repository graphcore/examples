# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
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


import poptorch
import popart


def build_ipu_config(args, seed=1234, gradient_accmulation=128, availble_memory_proportion=0.1):
    ipu_config = poptorch.Options()
    ipu_config.autoRoundNumIPUs(True)
    ipu_config.randomSeed(seed)
    ipu_config.Training.gradientAccumulation(gradient_accmulation)
    ipu_config.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    ipu_config.outputMode(poptorch.OutputMode.All)
    ipu_config.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))
    ipu_config.Precision.enableStochasticRounding(True)
    ipu_config._Popart.set("autoRecomputation", 3)
    ipu_config._Popart.set("disableGradAccumulationTensorStreams", True)
    ipu_config._Popart.set("outlineThreshold", 10.0)
    ipu_config.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    ipu_config._Popart.set(
        "accumulateOuterFragmentSettings.schedule", int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized)
    )
    ipu_config._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    ipu_config.setAvailableMemoryProportion(
        {f"IPU{i}": availble_memory_proportion for i in range(args.replication_factor * 2)}
    )
    ipu_config._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    ipu_config._Popart.setPatterns(
        {"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True}
    )
    ipu_config.enableExecutableCaching("exec")
    ipu_config.replicationFactor(args.replication_factor)
    ipu_config.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        .useOnChipStorage(not args.optimizer_state_offchip)
        .useReplicatedTensorSharding(args.replication_factor > 1)
    )

    return ipu_config
