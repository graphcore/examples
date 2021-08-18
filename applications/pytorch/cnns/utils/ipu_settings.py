# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import popart
from .logger import Logger


def inference_settings(opts, model_opts):
    if hasattr(opts, "opts.model_cache_path") and opts.model_cache_path is not None:
        model_opts.enableExecutableCaching(opts.model_cache_path)
    if opts.data == "synthetic":
        model_opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))
    partial_type = torch.float16 if opts.half_partial else torch.float32
    model_opts.Precision.setPartialsType(partial_type)
    # Use the faster GroupNorm implementation or compatible version.
    model_opts._Popart.set("groupNormStridedChannelGrouping", opts.enable_fast_groupnorm)
    engine_options = {}
    if opts.profile:
        engine_options = {
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": Logger.logdirname,
                "profiler.format": "v3",
                "autoReport.all": "true",
        }
        engine_options["opt.internalExchangeOptimisationTarget"] = opts.exchange_memory_target
    model_opts._Popart.set("engineOptions", engine_options)
    return model_opts


def train_settings(opts, model_opts):
    model_opts = inference_settings(opts, model_opts)
    model_opts.Precision.enableStochasticRounding(opts.enable_stochastic_rounding)

    model_opts.enableStableNorm(not opts.disable_stable_batchnorm)
    if opts.enable_fp_exceptions:
        model_opts._Popart.set("enableFloatingPointChecks", True)

    if opts.enable_recompute and len(opts.pipeline_splits) == 0:
        model_opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Standard))

    if opts.offload_optimizer:
        tensor_location = poptorch.TensorLocationSettings().useOnChipStorage(False)
        model_opts.TensorLocations.setOptimizerLocation(tensor_location)

    model_opts._Popart.set("disableGradAccumulationTensorStreams", True)

    num_stages = len(opts.pipeline_splits)+1
    if len(opts.available_memory_proportion) == 1:
        model_opts.setAvailableMemoryProportion({f'IPU{i}': opts.available_memory_proportion[0] for i in range(num_stages)})
    elif len(opts.available_memory_proportion) > 1:
            model_opts.setAvailableMemoryProportion({f'IPU{i}': amp for i, amp in enumerate(opts.available_memory_proportion)})


    if opts.disable_metrics:
        # if not interested in accurate metrics, return only subset of the predictions
        model_opts.anchorMode(poptorch.AnchorMode.Final)
    else:
        model_opts.anchorMode(poptorch.AnchorMode.All)

    return model_opts
