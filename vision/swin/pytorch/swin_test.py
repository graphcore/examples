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
import sys
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from config import get_config
from models.build import build_pipeline as build_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from optimizer import build_optimizer
from options import get_options
import poptorch
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from lr_scheduler import build_scheduler
from timm.models import resume_checkpoint
from dataset.ipu_mixup import Mixup
from pathlib import Path
import unittest
import pytest
import ctypes
import subprocess

swin_root_path = str(Path(__file__).parent)
sys.path.append(swin_root_path)


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, lab


def parse_option():
    parser = argparse.ArgumentParser("Swin Transformer training and evaluation script", add_help=False)
    parser.add_argument(
        "--cfg",
        default="SWIN_LARGE_224_22K_FINETUNE_1K_FP16_POD16",
        type=str,
        metavar="PATH",
        help="path to config file",
    )
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--num-workers", type=int, default=8, help="batch size for single GPU")
    parser.add_argument("--weights", type=str, help="weights for model")
    parser.add_argument("--device", type=str, default="", choices=["cpu", "ipu", "gpu"])
    parser.add_argument("--alignment", action="store_true", help="if alignment fwd or bwd")
    parser.add_argument("--half", default=True, action="store_true", help="use half")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Resume full model and optimizer state from checkpoint (default: none)",
    )
    parser.add_argument("--data-path", type=str, metavar="FILE", help="path to dataset")
    parser.add_argument("--checkpoint-output-dir", type=str, metavar="FILE", help="path to save output files")
    parser.add_argument("--pretrained-model", type=str, help="path to init checkpoint when fine tune models")
    parser.add_argument("--ga", type=int, help="Gradient Accumulations Steps")
    parser.add_argument("--amp", type=float, help="Available memory proportion")
    parser.add_argument("--rts", action="store_true", help="Replicated tensor sharding")
    parser.add_argument("--compile-only", action="store_true", help="Compile only")
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def get_random_datum(config):
    result = []
    batch_size = config.DATA.BATCH_SIZE * config.IPU.NUM_LOCALREPLICA * config.IPU.GRADIENT_ACCUMULATION_STEPS
    if config.PRECISION[0] == "half":
        use_half = True
    else:
        use_half = False
    dataset = GeneratedDataset(
        shape=[3, config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[0]], size=batch_size, half_precision=use_half
    )
    data = (dataset[i] for i in range(batch_size))
    for batches in zip(*data):
        result.append(torch.stack(batches))
    return result


class GeneratedDataset(Dataset):
    """
    Generated dataset creates a random dataset with the given shape and precision.
    The size determines the number of items in the dataset.
    """

    def __init__(self, shape, size=60000, half_precision=True):  # use_half
        self.size = size
        self.half_precision = half_precision
        self.data_shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        synthetic_data = torch.rand(self.data_shape)
        synthetic_label = torch.randint(0, 2, [1], dtype=torch.long)
        if self.half_precision:
            synthetic_data = synthetic_data.half()
        return synthetic_data, synthetic_label


def compile_model(poptorch_model, config):
    datum = get_random_datum(config)
    (pre_input, pre_label) = datum
    mixup_fn = Mixup(
        mixup_alpha=config.AUG.MIXUP,
        cutmix_alpha=config.AUG.CUTMIX,
        cutmix_minmax=config.AUG.CUTMIX_MINMAX,
        prob=config.AUG.MIXUP_PROB,
        switch_prob=config.AUG.MIXUP_SWITCH_PROB,
        mode=config.AUG.MIXUP_MODE,
        label_smoothing=config.MODEL.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES,
    )
    pre_input, pre_label = mixup_fn(pre_input, pre_label)
    poptorch_model.compile(pre_input, pre_label)
    return pre_input, pre_label


class TestSWIN(unittest.TestCase):
    @pytest.mark.ipus(8)
    def test_swin_model(self):
        cmd = "make"
        ret = subprocess.check_call(cmd, shell=True, cwd=swin_root_path)
        assert os.path.exists(os.path.join(swin_root_path, "custom_ops.so")), "please compile custom op"
        ctypes.cdll.LoadLibrary(os.path.join(swin_root_path, "custom_ops.so"))
        args, config = parse_option()
        config.defrost()
        config.IPU.NUM_LOCALREPLICA = 1
        config.IPU.GRADIENT_ACCUMULATION_STEPS = 16
        config.freeze()
        seed = config.SEED
        torch.manual_seed(seed)
        np.random.seed(seed)

        opts = get_options(config)

        self.train(args, opts, config)

    def train(self, args, opts, config):
        if config.AUG.MIXUP > 0.0:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        model = build_model(config=config, train_loss_fn=criterion)
        if args.half:
            print("use half")
            model.half()

        optimizer = build_optimizer(config, model)
        model = poptorch.trainingModel(model.train(), opts, optimizer=optimizer)
        data, targets = compile_model(model, config)
        lr_scheduler = build_scheduler(config, optimizer, 10008)
        start_epoch = 0
        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.step(start_epoch)
        for step in range(0, 100):
            time_s = time.time()
            model.setOptimizer(lr_scheduler.optimizer)
            _, loss = model(data, targets)
            time_e = time.time()
            print("batch time:", time_e - time_s)
            assert loss.mean() <= 10.0, "loss error"
