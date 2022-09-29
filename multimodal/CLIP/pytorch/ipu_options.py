# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import popart
import poptorch


def get_options(config):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''
    # Poptorch options
    opts = poptorch.Options()

    # Fix the random seeds
    opts.randomSeed(config.random_seed)
    np.random.seed(config.random_seed)

    opts.autoRoundNumIPUs(True)

    # Set device_iterations, replication and gradient accumulation factors
    opts.deviceIterations(config.device_iterations)
    opts.replicationFactor(config.replication_factor)
    opts.Training.gradientAccumulation(config.gradient_accumulation)

    if config.ipu_generate_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))

    # Return the final results from IPU to host
    opts.outputMode(poptorch.OutputMode.Final)

    opts.broadcastBuffers(False)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

    # Enable Replicated Tensor Sharding (RTS) of optimizer state with optimizer state residing either on-chip or in DRAM
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        # optimizer state lives on-chip
        .useOnChipStorage(config.state_onchip)
        # Shard optimizer state between replicas with zero-redundancy
        .useReplicatedTensorSharding(config.enable_rts)
        )

    opts.Precision.halfFloatCasting(poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)

    # Enable caching the compiled executable to disk
    if config.executable_cache_dir:
        opts.enableExecutableCaching(config.executable_cache_dir)

    # Use pipeline execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    # Set available transient memory for matmuls and convolutions operations
    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)


    # PopART options
    # Enable stochastic rounding (recommended for training with FP16)
    opts._Popart.set("enableStochasticRounding", True)
    opts._Popart.set("virtualGraphMode", int(popart.VirtualGraphMode.Manual))
    # Only stream needed tensors back to host
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("enableGradientAccumulation", True)
    opts._Popart.set("enableOutlining", True)
    opts._Popart.set("outlineThreshold", 10.0)
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    # Set the prefetch depth
    opts._Popart.set("defaultPrefetchBufferingDepth", 4)


    opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Pipeline))
    opts._Popart.set("decomposeGradSum", True)
    opts._Popart.set("batchSerializationSettings.batchSchedule", int(popart.BatchSerializationBatchSchedule.Isomorphic))


    # Half precision partials for matmuls and convolutions
    if config.enable_half_partials:
        opts._Popart.set("partialsTypeMatMuls", "half")
        opts._Popart.set("convolutionOptions", {'partialsType': "half"})

    engine_options = {"target.syncReplicasIndependently": "true",
                      "opt.maxComputeSetsPerLoweredCopy": "6"}

    opts._Popart.set("engineOptions", engine_options)

    return opts
