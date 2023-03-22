# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import argparse
import numpy as np
import torch
from config import get_config
from models.build import build_pipeline as build_model
import ctypes
from utils import AverageMeter

import poptorch
from dataset.build_ipu import build_dataloader_val, build_loader
from timm.utils import accuracy, AverageMeter
from timm.models import load_checkpoint


def set_opts():

    opts = poptorch.Options()
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(1)
    opts.replicationFactor(1)
    opts.Training.gradientAccumulation(1)
    opts.Precision.setPartialsType(torch.float)
    opts.randomSeed(42)
    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts.enableExecutableCaching("./cache")
    opts.outputMode(poptorch.OutputMode.All)
    opts.Precision.enableStochasticRounding(False)
    return opts


def parse_option():
    parser = argparse.ArgumentParser("Swin Transformer training and evaluation script", add_help=False)
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--data-path", type=str, required=True, metavar="FILE", help="path to dataset")
    parser.add_argument("--output", type=str, required=True, metavar="FILE", help="path to save output files")
    parser.add_argument(
        "--pretrained-model", type=str, default=None, help="path to init checkpoint when fine tune models"
    )
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--num-workers", type=int, default=8, help="batch size for single GPU")
    parser.add_argument("--weights", type=str, help="weights for model")
    parser.add_argument("--device", type=str, default="", choices=["cpu", "ipu", "gpu"])
    parser.add_argument("--alignment", action="store_true", help="if alignment fwd or bwd")
    parser.add_argument("--half", action="store_true", help="use half")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Resume full model and optimizer state from checkpoint (default: none)",
    )
    parser.add_argument("--checkpoint", default="", type=str, metavar="PATH", help="validate")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


@torch.no_grad()
def validate(data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        # compute output
        output = model(images)
        # measure accuracy and record loss
        loss = criterion(output.type(torch.float32), target)
        acc1, acc5 = accuracy(output.type(torch.float32), target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % 50 == 0:
            print(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
            )
    print(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def main():
    args, config = parse_option()
    opts = set_opts()
    dataset_val, data_loader_val = build_dataloader_val(config, opts)
    criterion = torch.nn.CrossEntropyLoss()
    model = build_model(config=config, train_loss_fn=criterion).eval()
    model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    model.eval()
    print(args.checkpoint)
    valid_opts = poptorch.Options()
    valid_opts.deviceIterations(256)
    valid_opts.outputMode(poptorch.OutputMode.All)
    model = poptorch.inferenceModel(model, valid_opts)
    load_checkpoint(model, args.checkpoint, strict=False)
    validate(data_loader_val, model)


if __name__ == "__main__":
    ctypes.cdll.LoadLibrary("./custom_ops.so")
    main()
