# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import yacs
import torch
import argparse
from tqdm import tqdm

import poptorch
from poptorch import inferenceModel, DataLoader, DataLoaderMode

from models.detector import Detector
from utils.tools import AutoAnchors
from utils.config import get_cfg_defaults
from utils.dataset import Dataset
from models.loss import PreprocessTargets


def ipu_options(cfg: yacs.config.CfgNode, model: Detector):
    device_iterations = cfg.ipuopts.device_iterations

    ipu_opts = poptorch.Options()
    ipu_opts.deviceIterations(device_iterations)
    ipu_opts.autoRoundNumIPUs(True)

    return ipu_opts


def find_optimal_anchors(opt, cfg, gen, image_size=896):
    print("Finding optimal anchors for image size " + str(image_size))
    cfg.model.image_size = image_size

    dataset = Dataset(opt.data, cfg, 'train')
    auto_anchors = AutoAnchors(dataset, cfg.model, gen=gen)
    new_anchors = auto_anchors()
    return new_anchors


def find_max_nlabels(opt, cfg, new_anchors):
    model = PreprocessTargets(cfg.model, new_anchors)

    ipu_opts = ipu_options(cfg, model)
    dataset = Dataset(opt.data, cfg, 'train')
    loader = DataLoader(ipu_opts,
                        dataset,
                        batch_size=cfg.model.micro_batch_size,
                        num_workers=cfg.system.num_workers,
                        mode=DataLoaderMode.Async)

    inference_model = inferenceModel(model.eval(), ipu_opts)

    max_nlabels = [torch.tensor([0]), ] * len(cfg.model.strides)
    pbar = tqdm(loader, desc='Finding the maximum number of labels after preprocessing')
    for _, (_, label, _, _) in enumerate(pbar):
        n_labels = inference_model(label)

        for j, old_max_nlabels in enumerate(max_nlabels):
            max_nlabels[j] = torch.maximum(n_labels[j], old_max_nlabels)

    return max_nlabels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='/localdata/datasets/', help='Dataset')
    parser.add_argument(
        '--config', type=str, default='configs/inference-yolov4p5.yaml', help='Configuration of the model')

    opt = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)
    cfg.model.auto_anchors = True

    image_sizes = [604, 640, 416]

    for image_size in image_sizes:
        anchors = find_optimal_anchors(opt, cfg, gen=1000, image_size=image_size)
        print("Optimal anchors are: ")
        print("p3 width: ", anchors[0].widths)
        print("p3 height: ", anchors[0].heights)
        print("p4 width: ", anchors[1].widths)
        print("p4 height: ", anchors[1].heights)
        print("p5 width: ", anchors[2].widths)
        print("p5 height: ", anchors[2].heights)

        max_nlabels = find_max_nlabels(opt, cfg, anchors)
        print("p3 max num labels: ", max_nlabels[0])
        print("p4 max num labels: ", max_nlabels[1])
        print("p5 max num labels: ", max_nlabels[2])
