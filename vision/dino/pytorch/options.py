# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import torch
import popart
import poptorch
import popdist


def alignment_options():
    opts = poptorch.Options()
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(1)
    opts.replicationFactor(1)
    opts.Training.gradientAccumulation(1)
    opts.randomSeed(42)
    opts.setExecutionStrategy(poptorch.ShardedExecution())
    return opts


def get_options(ga, pipeline=None, half=False, replic=1):
    opts = poptorch.Options()
    ipu_list = [0] if pipeline is None else pipeline
    mem_prop = {f'IPU{i}': 0.15 for i, _ in enumerate(ipu_list)}
    opts.randomSeed(42)
    opts.autoRoundNumIPUs(True)
    opts.Training.gradientAccumulation(ga)
    opts.Training.accumulationAndReplicationReductionType(
        poptorch.ReductionType.Sum)
    opts.deviceIterations(1)
    opts.replicationFactor(1)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(
            poptorch.AutoStage.SameAsIpu))
    opts.setAvailableMemoryProportion(mem_prop)
    if half:
        opts.Precision.setPartialsType(torch.half)
        opts.Precision.enableStochasticRounding(True)
    if replic > 1:
        opts.broadcastBuffers(False)
        # rts
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings().useReplicatedTensorSharding(
                True).minElementsForReplicatedTensorSharding(len(pipeline))
        )
    return opts


def train_options(
        use_popdist=False,
        ipu_per_replica=8,
        pipeline=None,
        ga=16,
        replic=1,
        di=1,
        synthetic_data=False,
        half=False,
        cachedir='./cachedir'):
    if use_popdist:
        opts = popdist.poptorch.Options(ipu_per_replica)
    else:
        opts = poptorch.Options()
        opts.replicationFactor(replic)

    opts.randomSeed(42)
    opts.enableExecutableCaching(cachedir)
    ipu_list = [0] if pipeline is None else pipeline
    mem_prop = {f'IPU{i}': 0.15 for i, _ in enumerate(ipu_list)}
    opts.autoRoundNumIPUs(True)
    opts.Training.gradientAccumulation(ga)
    opts.Training.accumulationAndReplicationReductionType(
        poptorch.ReductionType.Mean)
    opts.deviceIterations(di)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(
            poptorch.AutoStage.SameAsIpu))
    opts.setAvailableMemoryProportion(mem_prop)
    opts.outputMode(poptorch.OutputMode.All)
    if half:
        opts.Precision.setPartialsType(torch.half)
        opts.Precision.enableStochasticRounding(True)
    if replic > 1:
        # rts
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings().useReplicatedTensorSharding(
                True).minElementsForReplicatedTensorSharding(len(pipeline))
        )

    # Enable synthetic random data generated on device (so with no I/O)
    if synthetic_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))
    return opts
