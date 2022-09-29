# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

import popart
import poptorch
import torch


def get_opts(options: dict, modelType='training'):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/software.html#pytorch
    '''
    options = options.copy()
    options.update(options[modelType])

    opts = poptorch.Options()

    # Fix the random seeds
    opts.randomSeed(options.get('random_seed', 0))
    np.random.seed(options.get('random_seed', 0))
    torch.manual_seed(options.get('random_seed', 0))

    opts.autoRoundNumIPUs(options.get('autoRoundNumIPUs', False))

    # Set replication factors
    opts.replicationFactor(options.get('replication_factor', 1))

    # FloatingPointException
    opts.Precision.enableFloatingPointExceptions(
        options.get('enableFloatingPointExceptions', True))

    # Enable caching the compiled executable to disk
    executable_cache_dir = options.get('executable_cache_dir', "./exps/exe_cache")
    if executable_cache_dir:
        opts.enableExecutableCaching(executable_cache_dir)

    # Use pipeline execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Pipeline))

    # Set available transient memory for matmuls and convolutions operations
    available_memory_proportion = options.get('available_memory_proportion', {})
    if len(available_memory_proportion):
        opts.setAvailableMemoryProportion(available_memory_proportion)

    opts._Popart.set("accumulateOuterFragmentSettings.schedule", int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))

    # Half precision partials for matmuls and convolutions
    enable_half_partials = options.get('enable_half_partials', True)
    if enable_half_partials:
        opts._Popart.set("partialsTypeMatMuls", "half")
        opts._Popart.set("convolutionOptions", {'partialsType': "half"})
        opts.Precision.setPartialsType(torch.half)

    # PopART options
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    # Enable stochastic rounding (recommended for training with FP16)
    opts._Popart.set("enableStochasticRounding",
                     options.get('enableStochasticRounding', True))

    # Half precision partials for matmuls and convolutions
    enable_half_partials = options.get('enable_half_partials', True)
    if enable_half_partials:
        opts._Popart.set("partialsTypeMatMuls", "half")
        opts._Popart.set("convolutionOptions", {'partialsType': "half"})
        opts.Precision.setPartialsType(torch.half)

    engine_options = {"target.syncReplicasIndependently": "true",
                      "opt.maxComputeSetsPerLoweredCopy": "4"}
    opts._Popart.set("engineOptions", engine_options)

    if modelType == 'training':
        opts.deviceIterations(options['training'].get('batches_per_step', 8))
        opts.outputMode(poptorch.OutputMode.Final)
        opts.Training.gradientAccumulation(
            options['training'].get('gradientAccumulation', 15))
        opts._Popart.set("enableGradientAccumulation",
                         options.get('enableGradientAccumulation', True))
        enable_rts = options.get('enable_rts', True)
        # Enable Replicated Tensor Sharding (RTS) of optimizer state with optimizer state residing either on-chip or in DRAM
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings()
            # Optimizer state lives on-chip
            .useOnChipStorage(False)  # Put the optimizer in remote buffer
            # Shard optimizer state between replicas with zero-redundancy
            # .useReplicatedTensorSharding(enable_rts)
        )
    if modelType == 'inference':
        opts.deviceIterations(options['inference'].get('batches_per_step', 8))
        # return all results from IPU to host
        opts.outputMode(poptorch.OutputMode.All)

    return opts
