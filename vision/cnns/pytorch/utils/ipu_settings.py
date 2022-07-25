# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import popart
import logging
from .logger import Logger


def inference_settings(args, opts):
    if hasattr(args, "model_cache_path") and args.model_cache_path is not None:
        opts.enableExecutableCaching(args.model_cache_path)
    if args.data == "synthetic":
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))
    partial_type = torch.float16 if args.half_partial else torch.float32
    opts.Precision.setPartialsType(partial_type)
    # Use the faster GroupNorm implementation or compatible version.
    opts._Popart.set("groupNormStridedChannelGrouping", args.enable_fast_groupnorm)
    engine_options = {"target.deterministicWorkers": "portable"}  # avoid replica weight drift
    if args.profile and (not(hasattr(args, "popdist_rank")) or args.popdist_rank == 0):
        logging.info(f"Profile files will be available in {Logger.logdirname}.")
        engine_options = {
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": Logger.logdirname,
                "profiler.format": "v3",
                "autoReport.all": "true",
        }
    if args.exchange_memory_target is not None:
        engine_options["opt.internalExchangeOptimisationTarget"] = args.exchange_memory_target
    if len(engine_options) > 0:
        opts._Popart.set("engineOptions", engine_options)
    if args.num_io_tiles > 0:
        opts.setExecutionStrategy(poptorch.ShardedExecution())
        opts._Popart.set("defaultPrefetchBufferingDepth", 3)
        opts.TensorLocations.numIOTiles(args.num_io_tiles)

    # Use offline IPU for compilation only, it needs an appropriate version.
    if args.compile_only:
        opts.useOfflineIpuTarget(poptorch.ipuHardwareVersion())

    return opts


def train_settings(args, opts):
    opts = inference_settings(args, opts)
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts.Precision.enableStochasticRounding(args.enable_stochastic_rounding)

    opts.enableStableNorm(not args.disable_stable_batchnorm)
    opts.Precision.enableFloatingPointExceptions(args.enable_fp_exceptions)

    if not(args.recompute_mode == "none") and len(args.pipeline_splits) == 0:
        opts._Popart.set("explicitRecomputation", True)
        if args.recompute_mode == "auto":
            opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Standard))
        elif args.recompute_mode == "manual":
            opts._Popart.set("autoRecomputation", int(popart.RecomputationType.RecomputeAll))

    if args.offload_optimizer or args.enable_optimizer_rts:
        location_settings = poptorch.TensorLocationSettings()
        location_settings.useOnChipStorage(not args.offload_optimizer)
        location_settings.useReplicatedTensorSharding(args.enable_optimizer_rts)
        opts.TensorLocations.setOptimizerLocation(location_settings)

    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("disableOptimizerStateTensorStreams", True)

    num_stages = len(args.pipeline_splits)+1
    if len(args.available_memory_proportion) == 1:
        opts.setAvailableMemoryProportion({f'IPU{i}': args.available_memory_proportion[0] for i in range(num_stages)})
    elif len(args.available_memory_proportion) > 1:
        opts.setAvailableMemoryProportion({f'IPU{i}': amp for i, amp in enumerate(args.available_memory_proportion)})

    if args.auto_loss_scaling:
        opts.Training.setAutomaticLossScaling(True)

    opts.outputMode(poptorch.OutputMode.Sum)
    return opts
