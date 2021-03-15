# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import poptorch
import popart
from .logger import Logger


def inference_settings(opts, model_opts):
    if opts.model_cache_path is not None:
        model_opts.enableExecutableCaching(opts.model_cache_path)
    if opts.data == "synthetic":
        model_opts.Popart.set("syntheticDataMode", int(popart.SyntheticDataMode.RandomNormal))
    if opts.half_partial:
        model_opts.Popart.set("partialsTypeMatMuls", "half")
        model_opts.Popart.set("convolutionOptions", {'partialsType': 'half'})

    if opts.profile:
        engine_options = {
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": Logger.logdirname,
                "profiler.format": "v3",
                "autoReport.all": "true",
        }
        model_opts.Popart.set("engineOptions", engine_options)
    return model_opts


def train_settings(opts, model_opts):
    model_opts = inference_settings(opts, model_opts)
    model_opts.Popart.set("enableStableNorm", True)

    model_opts.Popart.set("enableStochasticRounding", opts.enable_stochastic_rounding)
    if opts.data == "synthetic":
        model_opts.Popart.set("syntheticDataMode", int(popart.SyntheticDataMode.RandomNormal))
    if opts.half_partial:
        model_opts.Popart.set("partialsTypeMatMuls", "half")
        model_opts.Popart.set("convolutionOptions", {'partialsType': 'half'})

    if opts.enable_fp_exceptions:
        model_opts._Popart.set("enableFloatingPointChecks", True)

    if opts.enable_recompute and len(opts.pipeline_splits) == 0:
            model_opts.Popart.set("autoRecomputation", int(popart.RecomputationType.Standard))

    if opts.offload_optimizer:
        tensor_location = poptorch.TensorLocationSettings().useOnChipStorage(False)
        model_opts.TensorLocations.setOptimizerLocation(tensor_location)

    model_opts.Popart.set("disableGradAccumulationTensorStreams", True)

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
