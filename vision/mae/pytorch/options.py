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

import torch
import poptorch
import popdist
import popart
from util.log import logger


def alignment_options(ipus=8):
    opts = poptorch.Options()

    opts.autoRoundNumIPUs(True)
    opts.Training.gradientAccumulation(1)
    opts.deviceIterations(1)
    opts.replicationFactor(1)
    opts.randomSeed(2)
    mem_prop = {f'IPU{i}': 0.15 for i in range(ipus)}

    opts.setExecutionStrategy(poptorch.ShardedExecution())
    opts.setAvailableMemoryProportion(mem_prop)
    return opts


def train_options(
        use_popdist,
        gradient_accumulation_count=8,
        replica=1,
        iteration=1,
        half=False,
        als=False,
        ipu_per_replica=4,
        rts=False):

    if use_popdist:
        opts = popdist.poptorch.Options(ipu_per_replica)
    else:
        opts = poptorch.Options()
        opts.replicationFactor(replica)

    opts.randomSeed(42)
    opts.enableExecutableCaching('./cachedir')
    mem_prop = {f'IPU{i}': 0.15 for i in range(ipu_per_replica)}

    opts.autoRoundNumIPUs(True)
    opts.Training.gradientAccumulation(gradient_accumulation_count)
    opts.Training.accumulationAndReplicationReductionType(
        poptorch.ReductionType.Mean)

    opts.Training.setMeanAccumulationAndReplicationReductionStrategy(
        poptorch.MeanReductionStrategy.Running
    )
    opts.deviceIterations(iteration)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(
            poptorch.AutoStage.AutoIncrement))
    opts.setAvailableMemoryProportion(mem_prop)

    if half:
        opts.Precision.enableStochasticRounding(True)
        opts.Precision.setPartialsType(torch.half)
        if als:
            opts.Training.setAutomaticLossScaling(True)
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set(
        "replicatedCollectivesSettings.prepareScheduleForMergingCollectives",
        True)
    opts._Popart.set(
        "replicatedCollectivesSettings.mergeAllReduceCollectives",
        True)

    # rts
    if rts:
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings().useReplicatedTensorSharding(
                True).minElementsForReplicatedTensorSharding(ipu_per_replica)
        )

    return opts


def finetune_options(gradient_accumulation_count=8, replica=1, device_iterations=1,
                     half=False, ipu_per_replica=8, opt_type='train'):

    if popdist.isPopdistEnvSet():
        opts = popdist.poptorch.Options(ipu_per_replica)
    else:
        opts = poptorch.Options()
        opts.replicationFactor(replica)
    opts.randomSeed(42)
    opts.enableExecutableCaching('./cachedir')
    logger.info(f'ipu_per_replica {ipu_per_replica}')
    mem_prop = {f'IPU{i}': 0.12 for i in range(ipu_per_replica)}
    opts.autoRoundNumIPUs(True)
    if opt_type == 'train':
        opts.deviceIterations(device_iterations)
        opts.Training.gradientAccumulation(gradient_accumulation_count)
        opts.Training.accumulationAndReplicationReductionType(
            poptorch.ReductionType.Mean)
        opts.setExecutionStrategy(
            poptorch.PipelinedExecution(
                poptorch.AutoStage.SameAsIpu))
    else:
        opts.deviceIterations(device_iterations)
        opts.setExecutionStrategy(poptorch.ShardedExecution())
        logger.info('poptorch.ShardedExecution')
    opts.setAvailableMemoryProportion(mem_prop)
    if half:
        opts.Precision.enableStochasticRounding(True)
        opts.Precision.setPartialsType(torch.half)

    # rts
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useReplicatedTensorSharding(
            True).minElementsForReplicatedTensorSharding(ipu_per_replica)
    )

    return opts
