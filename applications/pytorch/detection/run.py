# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import logging
import time
from tqdm import tqdm
import yacs

import poptorch
from poptorch import trainingModel, inferenceModel, DataLoader, DataLoaderMode
import torch
from torch.utils.data import DataLoader as torchDataLoader

from models.detector import Detector
from models.yolov4_p5 import Yolov4P5
from utils.config import get_cfg_defaults, override_cfg, save_cfg
from utils.dataset import Dataset
from utils.parse_args import parse_args
from utils.tools import load_and_fuse_pretrained_weights, post_processing, StatRecorder
from utils.visualization import plotting_tool

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger('Detector')


def ipu_options(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode, model: Detector):
    """Configurate the IPU options using cfg and opt options.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
        model[Detector]: a torch Detector Model
    Returns:
        ipu_opts: Options for the IPU configuration
    """
    batches_per_step = cfg.ipuopts.batches_per_step
    half = cfg.model.half

    ipu_opts = poptorch.Options()
    ipu_opts.deviceIterations(batches_per_step)
    ipu_opts.autoRoundNumIPUs(True)

    if opt.benchmark:
        ipu_opts.Distributed.disable()

    if half:
        ipu_opts.Precision.setPartialsType(torch.float16)
        model.half()

    return ipu_opts


def get_loader(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode, ipu_opts: poptorch.Options):
    """Gets a new loader for the model.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
        ipu_opts: Options for the IPU configuration
    Returns:
        model[Detector]: a torch Detector Model
    """
    dataset = Dataset(path=opt.data, cfg=cfg)

    # Creates a loader using the dataset
    if cfg.model.ipu:
        loader = DataLoader(ipu_opts,
                            dataset,
                            batch_size=cfg.model.micro_batch_size,
                            num_workers=cfg.system.num_workers,
                            mode=DataLoaderMode.Async)
    else:
        loader = torchDataLoader(dataset, batch_size=cfg.model.micro_batch_size)

    return loader


def get_model_and_loader(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode):
    """Prepares the model and gets a new loader for the model.
    Parameters:
        opt: opt object containing options introduced in the command line
        cfg: yacs object containing the config
    Returns:
        model[Detector]: a torch Detector Model
        loader[DataLoader]: a torch or poptorch DataLoader containing the specified dataset on "cfg"
    """

    # Create model
    model = Yolov4P5(cfg)

    if cfg.model.mode == "train":
        model.train()
    else:
        model.eval()

        # Load weights and fuses some batch normalizations with some convolutions
        if cfg.model.normalization == 'batch':
            if opt.weights:
                print("loading pretrained weights")
                model = load_and_fuse_pretrained_weights(model, opt)
            model.optimize_for_inference()

    # Create the specific ipu options if cfg.model.ipu
    ipu_opts = ipu_options(opt, cfg, model) if cfg.model.ipu else None

    # Creates the loader
    loader = get_loader(opt, cfg, ipu_opts)

    # Calls the poptorch wrapper and compiles the model
    if cfg.model.ipu:
        if cfg.model.mode == "train":
            model = trainingModel(model, ipu_opts)
        else:
            model = inferenceModel(model, ipu_opts)
        try:
            img, _, _, _ = next(iter(loader))
            model.compile(img)
            warm_up_iterations = 100
            for _ in range(warm_up_iterations):
                _ = model(img)
        except Exception as e:
            print(e.args)
            exit(0)

    return model, loader


def inference(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode):
    model, loader = get_model_and_loader(opt, cfg)

    stat_recorder = StatRecorder(cfg)

    train_progress = tqdm(loader)
    train_progress.set_description("Running inference")
    for batch_idx, (transformed_images, transformed_labels, image_sizes, image_indxs) in enumerate(train_progress):
        start_time = time.time()
        y = model(transformed_images)
        inference_step_time = time.time() - start_time

        inference_round_trip_time = model.getLatency() if cfg.model.ipu else (inference_step_time,) * 3  # returns (min, max, avg) latency

        processed_batch = post_processing(cfg, y, image_sizes, transformed_labels)

        stat_recorder.record_inference_stats(inference_round_trip_time, inference_step_time)

        if cfg.inference.plot_output and batch_idx % cfg.inference.plot_step == 0:
            plotting_tool(cfg, processed_batch[0], [loader.dataset.get_image(img_idx) for img_idx in image_indxs])

        if opt.benchmark and batch_idx == 100:
            break

        pruned_preds_batch = processed_batch[0]
        processed_labels_batch = processed_batch[1]
        if cfg.eval.metrics:
            for idx, (pruned_preds, processed_labels) in enumerate(zip(pruned_preds_batch, processed_labels_batch)):
                stat_recorder.record_eval_stats(processed_labels, pruned_preds, image_sizes[idx])

    stat_recorder.logging(print)


if __name__ == "__main__":
    opt = parse_args()
    if len(opt.data) > 0. and opt.data[-1] != '/':
        opt.data += '/'

    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)
    cfg = override_cfg(opt, cfg)
    cfg.freeze()
    save_cfg('./configs/override-inference-yolov4p5.yaml', cfg)

    if opt.show_config:
        logger.info(f"Model options: \n'{cfg}'")

    inference(opt, cfg)
