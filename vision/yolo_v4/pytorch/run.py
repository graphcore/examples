# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
import argparse
from datetime import datetime
import logging
import math
import numpy as np
from pathlib import Path
import os
import time
from tqdm import tqdm
from typing import Callable, Union
import yacs
import wandb

import poptorch
from poptorch import trainingModel, inferenceModel, DataLoader, DataLoaderMode
from poptorch.optim import Optimizer, SGD
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader as torchDataLoader
from torchinfo import summary

from models.detector import Detector
from models.yolov4_p5 import Yolov4P5
from models.recomputation import recomputation_checkpoint
from utils.config import get_cfg_defaults, override_cfg, save_cfg
from utils.dataset import Dataset
from utils.parse_args import parse_args
from utils.postprocessing import post_processing, IPUPredictionsPostProcessing
from utils.tools import load_and_fuse_pretrained_weights, StatRecorder
from utils.visualization import plotting_tool
from utils.weight_avg import average_model_weights


path_to_detection = Path(__file__).parent.resolve()
os.environ["PYTORCH_APPS_DETECTION_PATH"] = str(path_to_detection)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger("Detector")


def ipu_options(cfg: yacs.config.CfgNode, model: Detector, mode: str, opt: argparse.ArgumentParser = None):
    """Configurate the IPU options using cfg and opt options.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
        model[Detector]: a torch Detector Model
        mode: str indicating 'train', 'test' or 'test_inference'
    Returns:
        ipu_opts: Options for the IPU configuration
    """
    device_iterations = cfg.ipuopts.device_iterations

    ipu_opts = poptorch.Options()
    ipu_opts.deviceIterations(device_iterations)
    ipu_opts.autoRoundNumIPUs(True)
    # Calculate the number of replicas from the pipeline length
    ipu_opts.replicationFactor(cfg.system.num_ipus // (1 + len(cfg.model.pipeline_splits)))

    ipu_opts.enableExecutableCaching(cfg.training.exec_cache_path)

    # Compile offline (no IPUs required)
    if opt is not None and opt.compile_only:
        ipu_opts.useOfflineIpuTarget()

    if opt is not None and opt.profile_dir:
        ipu_opts.enableProfiling(opt.profile_dir)

    if cfg.ipuopts.available_memory_proportion:
        amp = cfg.ipuopts.available_memory_proportion
        if isinstance(amp, float):
            amp_dict = {f"IPU{i}": amp for i in range(cfg.system.num_ipus)}
        elif isinstance(cfg.ipuopts.available_memory_proportion, list):
            assert len(amp) == len(cfg.model.pipeline_splits) + 1
            amp_dict = {f"IPU{i}": value for i, value in enumerate(amp)}
        else:
            raise TypeError("Wrong type of cfg.ipuopts.available_memory_proportion. " "Use either float or list.")
        ipu_opts.setAvailableMemoryProportion(amp_dict)

    if opt is not None and opt.benchmark:
        ipu_opts.Distributed.disable()

    if cfg.model.precision == "half":
        ipu_opts.Precision.setPartialsType(torch.float16)
        model.half()
    elif cfg.model.precision == "mixed":
        ipu_opts.Precision.setPartialsType(torch.float16)
        model.half()
        model.headp3 = model.headp3.float()
        model.headp4 = model.headp4.float()
        model.headp5 = model.headp5.float()
    elif cfg.model.precision != "single":
        raise ValueError("Only supoprt half, mixed or single precision")

    if mode == "train":
        ipu_opts.Training.gradientAccumulation(cfg.ipuopts.gradient_accumulation)
        ipu_opts.outputMode(poptorch.OutputMode.Sum)
        ipu_opts.Training.setAutomaticLossScaling(enabled=cfg.training.auto_loss_scaling)
        ipu_opts.Precision.enableStochasticRounding(cfg.training.stochastic_rounding)

        if cfg.model.sharded:
            ipu_opts.setExecutionStrategy(poptorch.ShardedExecution())
        else:
            ipu_opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    return ipu_opts


def get_loader(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode, ipu_opts: poptorch.Options, mode: str):
    """Gets a new loader for the model.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
        ipu_opts: Options for the IPU configuration
        mode: str indicating 'train', 'test' or 'test_inference'
    Returns:
        model[Detector]: a torch Detector Model
    """
    dataset = Dataset(path=opt.data, cfg=cfg, mode=mode)
    shuffle = mode == "train"
    # Creates a loader using the dataset
    if cfg.model.ipu:
        loader = DataLoader(
            ipu_opts,
            dataset,
            batch_size=cfg.model.micro_batch_size,
            shuffle=shuffle,
            num_workers=cfg.system.num_workers,
            mode=DataLoaderMode.Async,
        )
    else:
        loader = torchDataLoader(dataset, batch_size=cfg.model.micro_batch_size, shuffle=shuffle)

    return loader


def get_optimizer(cfg: yacs.config.CfgNode, model: Detector):
    """Returns the correct type of optimizer for ipu/cpu training
    Parameters:
        cfg: yacs object containing the config
        model: a Detector object
    Returns:
        optimizer: a torch/poptorch optimizer
    """
    nominal_batch_size = 64
    weight_decay = (
        cfg.training.weight_decay * cfg.model.micro_batch_size * cfg.ipuopts.gradient_accumulation / nominal_batch_size
    )
    param_group0, param_group1, param_group2 = [], [], []
    for k, v in model.named_parameters():
        v.requires_grad = True
        if ".bias" in k:
            param_group2.append(v)
        elif ".weight" in k and ".norm" not in k:
            param_group1.append(v)
        else:
            param_group0.append(v)
    if cfg.model.ipu:
        optimizer = SGD(
            param_group0,
            lr=cfg.training.initial_lr,
            momentum=cfg.training.momentum,
            loss_scaling=cfg.training.loss_scaling,
            use_combined_accum=True,
            dampening=0,
        )
    else:
        optimizer = SGD(param_group0, lr=cfg.training.initial_lr, momentum=cfg.training.momentum)

    optimizer.add_param_group({"params": param_group1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": param_group2})
    return optimizer


def get_model_and_loader(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode, mode: str):
    """Prepares the model and gets a new loader for the model.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
        mode: str indicating 'train', 'test' or 'test_inference'
    Returns:
        model[Detector]: a torch Detector Model
        loader[DataLoader]: a torch or poptorch DataLoader containing the specified dataset on "cfg"
    """

    # Create model
    model = Yolov4P5(cfg)

    # Insert the pipeline splits if using pipeline
    if cfg.model.pipeline_splits:
        named_layers = {name: layer for name, layer in model.named_modules()}
        for ipu_idx, split in enumerate(cfg.model.pipeline_splits):
            named_layers[split] = poptorch.BeginBlock(ipu_id=ipu_idx + 1, layer_to_call=named_layers[split])

    if len(cfg.model.recomputation_ckpts):
        for name, layer in model.named_modules():
            if name in cfg.model.recomputation_ckpts:
                recomputation_checkpoint(layer)

    # Load weights and fuses some batch normalizations with some convolutions
    if cfg.model.normalization == "batch":
        if opt.weights:
            print("loading pretrained weights")
            model = load_and_fuse_pretrained_weights(model, opt.weights, mode != "train")

    if mode == "train":
        model.train()
    else:
        model.optimize_for_inference()
        model.eval()

    if opt.print_summary:
        summary(
            model,
            input_size=(
                cfg.model.micro_batch_size,
                cfg.model.input_channels,
                cfg.model.image_size,
                cfg.model.image_size,
            ),
        )
        print("listing all layers by names")
        named_layers = {name: layer for name, layer in model.named_modules()}
        for layer in named_layers:
            print(layer)

    # Create the specific ipu options if cfg.model.ipu
    ipu_opts = ipu_options(cfg, model, mode, opt) if cfg.model.ipu else None

    # Creates the loader
    loader = get_loader(opt, cfg, ipu_opts, mode)

    # Calls the poptorch wrapper and compiles the model
    if cfg.model.ipu:
        img, labels, _, _ = next(iter(loader))
        if cfg.model.mode == "train":
            optimizer = get_optimizer(cfg, model)
            model = trainingModel(model, ipu_opts, optimizer=optimizer)
        else:
            model = inferenceModel(model, ipu_opts)
            model.compile(img)
            if opt.benchmark:
                warm_up_iterations = 100
                for _ in range(warm_up_iterations):
                    _ = model(img)

    return model, loader


def set_warmup_lr_and_momentum(
    optimizer: Optimizer,
    num_warmup: int,
    batch_idx: int,
    lr_lambda: Callable[[int], float],
    epoch: int,
    momentum: float,
):
    xi = [0, num_warmup]
    for j, x in enumerate(optimizer.param_groups):
        x["lr"] = np.interp(batch_idx, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lr_lambda(epoch + 1)])
        if "momentum" in x:
            x["momentum"] = np.interp(batch_idx, xi, [0.9, momentum])
    return optimizer


def inference(
    opt: argparse.ArgumentParser,
    cfg: yacs.config.CfgNode,
    model: Union[Detector, poptorch.PoplarExecutor],
    loader: Union[torchDataLoader, DataLoader],
    stat_recorder: StatRecorder,
    run_coco_eval: bool = False,
):
    inference_progress = tqdm(loader)
    inference_progress.set_description("Running inference")
    stat_recorder.reset_eval_stats()
    for batch_idx, (transformed_images, transformed_labels, image_sizes, image_indxs) in enumerate(inference_progress):
        start_time = time.time()
        y = model(transformed_images)
        inference_step_time = time.time() - start_time

        inference_round_trip_time = (
            model.getLatency() if cfg.model.ipu else (inference_step_time,) * 3
        )  # returns (min, max, avg) latency

        processed_batch = post_processing(cfg, y, image_sizes, transformed_labels)

        stat_recorder.record_inference_stats(inference_round_trip_time, inference_step_time)

        if cfg.inference.plot_output and batch_idx % cfg.inference.plot_step == 0:
            img_paths = plotting_tool(
                cfg, processed_batch[0], [loader.dataset.get_image(img_idx) for img_idx in image_indxs]
            )
            if opt.wandb:
                wandb.log({"inference_batch_{}".format(batch_idx): [wandb.Image(path) for path in img_paths]})

        if opt.benchmark and batch_idx == 100:
            break

        pruned_preds_batch = processed_batch[0]
        processed_labels_batch = processed_batch[1]
        if cfg.eval.metrics:
            for idx, (pruned_preds, processed_labels) in enumerate(zip(pruned_preds_batch, processed_labels_batch)):
                stat_recorder.record_eval_stats(
                    processed_labels,
                    pruned_preds,
                    image_sizes[idx],
                    loader.dataset.images_id[image_indxs[idx]],
                    run_coco_eval,
                )

    stat_recorder.logging(print, run_coco_eval)


if __name__ == "__main__":
    opt = parse_args()
    if len(opt.data) > 0.0 and opt.data[-1] != "/":
        opt.data += "/"

    cfg = get_cfg_defaults()

    cfg.merge_from_file(opt.config)
    cfg = override_cfg(opt, cfg)
    cfg.freeze()
    config_filename = Path(opt.config)
    config_filename = config_filename.with_name(f"override-{config_filename.name}")
    save_cfg(config_filename, cfg)

    if opt.show_config:
        logger.info(f"Model options: \n'{cfg}'")

    stat_recorder = StatRecorder(cfg, opt.data, opt.wandb)
    model, loader = get_model_and_loader(opt, cfg, cfg.model.mode)

    if opt.compile_only:
        logger.info("Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
        sys.exit(0)

    if cfg.model.mode == "train":
        logger.error("Training to be implemented!")
    else:
        run_coco_eval = cfg.eval.metrics and not opt.benchmark
        inference(opt, cfg, model, loader, stat_recorder, run_coco_eval)
