# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import io
import numpy as np
import os
from PIL import Image
import requests
import yacs

import torch
import poptorch
from poptorch import inferenceModel, trainingModel, Options

from models.detector import Detector
from models.yolov4_p5 import Yolov4P5
from utils.config import get_cfg_defaults
from utils.preprocessing import ResizeImage, Pad, ToTensor
from utils.postprocessing import post_processing
from utils.tools import load_and_fuse_pretrained_weights, StatRecorder


def get_cfg():
    cfg = get_cfg_defaults()

    cfg.model.image_size = 416
    cfg.inference.nms = True
    cfg.inference.class_conf_threshold = 0.001
    cfg.inference.iou_threshold = 0.65
    cfg.inference.nms_max_detections = 10
    cfg.inference.pre_nms_topk_k = 1180
    cfg.ipuopts.device_iterations = 1
    cfg.model.normalization = "batch"
    cfg.model.activation = "mish"
    cfg.model.half = False
    cfg.model.uint_io = True
    cfg.model.input_channels = 3
    cfg.model.micro_batch_size = 1
    cfg.model.mode = "test"
    cfg.model.ipu = True

    return cfg


def ipu_options(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode, model: Detector):
    """Configurate the IPU options using cfg and opt options.
        Parameters:
            opt: opt object containing options introduced in the command line
            cfg: yacs object containing the config
            model[Detector]: a torch Detector Model
        Returns:
            ipu_opts: Options for the IPU configuration
    """
    device_iterations = cfg.ipuopts.device_iterations
    half = cfg.model.half

    ipu_opts = Options()
    ipu_opts.deviceIterations(device_iterations)
    ipu_opts.autoRoundNumIPUs(True)

    if half:
        ipu_opts.Precision.setPartialsType(torch.float16)
        model.half()

    if cfg.model.mode == 'train':
        ipu_opts.Training.gradientAccumulation(cfg.ipuopts.gradient_accumulation)

        if cfg.model.sharded:
            ipu_opts.setExecutionStrategy(poptorch.ShardedExecution())
        else:
            ipu_opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    return ipu_opts


def get_image_and_label(cfg):
    url_sample_image = 'http://images.cocodataset.org/val2017/000000100238.jpg'
    img_data = requests.get(url_sample_image).content
    image = Image.open(io.BytesIO(img_data)).convert('RGB')

    height, width = image.size
    image_sizes = torch.Tensor([[height, width]])

    label = np.array([[39, 0.319508, 0.745573, 0.020516, 0.028479],
                      [0, 0.484391, 0.583271, 0.360031, 0.833458],
                      [0, 0.685664, 0.494917, 0.284422, 0.986458],
                      [0, 0.869086, 0.720719, 0.207766, 0.549563],
                      [0, 0.168453, 0.526521, 0.333531, 0.914208],
                      [29, 0.166422, 0.562135, 0.118313, 0.139687],
                      [29, 0.480703, 0.565990, 0.135906, 0.120813],
                      [26, 0.591977, 0.203583, 0.045234, 0.121958],
                      [26, 0.349672, 0.619479, 0.150000, 0.568833],
                      [29, 0.708734, 0.284302, 0.118188, 0.159854]])

    if cfg.model.mode == 'train':
        label[:, 0] = label[:, 0] + 1
    resizer = ResizeImage(cfg.model.image_size)
    padder = Pad(cfg.model.image_size)
    to_tensor = ToTensor(int(cfg.dataset.max_bbox_per_scale), "uint")

    item = (image, label)
    item = resizer(item)
    image, label = padder(item)
    image, label = to_tensor((np.array(image), label))

    return image.unsqueeze(axis=0), label.unsqueeze(axis=0), image_sizes


def prepare_model(cfg, debugging_nms=False):
    opt = argparse.ArgumentParser()
    torch.manual_seed(100)
    model = Yolov4P5(cfg, debugging_nms=debugging_nms)

    if cfg.model.mode == 'test':
        opt.weights = os.environ['PYTORCH_APPS_DETECTION_PATH'] + '/weights/yolov4-p5-sd.pt'
        model.eval()
        model = load_and_fuse_pretrained_weights(model, opt.weights)
        model.optimize_for_inference()
    else:
        model.train()

    if cfg.model.ipu:
        ipu_opts = ipu_options(opt, cfg, model)
        if cfg.model.pipeline_splits:
            named_layers = {name: layer for name, layer in model.named_modules()}
            for ipu_idx, split in enumerate(cfg.model.pipeline_splits):
                named_layers[split] = poptorch.BeginBlock(ipu_id=ipu_idx + 1, layer_to_call=named_layers[split])
        if cfg.model.mode == 'test':
            return inferenceModel(model, ipu_opts)
        else:
            optimizer = poptorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, loss_scaling=1, use_combined_accum=True)
            return trainingModel(model, ipu_opts, optimizer)
    else:
        return model


def post_process_and_eval(cfg, y, image_sizes, transformed_labels):
    stat_recorder = StatRecorder(cfg, '')
    processed_batch = post_processing(cfg, y, image_sizes, transformed_labels)

    pruned_preds_batch = processed_batch[0]
    processed_labels_batch = processed_batch[1]
    for idx, (pruned_preds, processed_labels) in enumerate(zip(pruned_preds_batch, processed_labels_batch)):
        stat_recorder.record_eval_stats(processed_labels, pruned_preds, image_sizes[idx], '100238', False)

    return stat_recorder.compute_and_print_eval_metrics(print)
