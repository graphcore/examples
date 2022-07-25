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
import pdb
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from lr_scheduler import build_scheduler
from timm.models import resume_checkpoint
import unittest
import pytest


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, lab


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script',
        add_help=False)
    parser.add_argument(
        '--cfg',
        default='./configs/swin_base_linear.yaml',
        type=str,
        metavar='PATH',
        help='path to config file',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help="batch size for single GPU")
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help="batch size for single GPU")
    parser.add_argument('--weights', type=str, help='weights for model')
    parser.add_argument(
        '--device',
        type=str,
        default='',
        choices=[
            'cpu',
            'ipu',
            'gpu'])
    parser.add_argument(
        '--alignment',
        action='store_true',
        help='if alignment fwd or bwd')
    parser.add_argument(
        '--half',
        default=True,
        action='store_true',
        help='use half')
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='Resume full model and optimizer state from checkpoint (default: none)')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def get_random_datum():
    result = []
    dataset = GeneratedDataset(shape=[3, 224, 224], size=128,  # 1024 for global batch
                               half_precision=True)
    data = (dataset[i] for i in range(128))
    for batches in zip(*data):
        result.append(torch.stack(batches))
    return result


class GeneratedDataset(Dataset):
    """
    Generated dataset creates a random dataset with the given shape and precision.
    The size determines the number of items in the dataset.
    """

    def __init__(self, shape, size=60000, half_precision=True):
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


class TestSWIN(unittest.TestCase):

    @pytest.mark.ipus(2)
    def test_swin_model(self):
        args, config = parse_option()
        seed = config.SEED
        torch.manual_seed(seed)
        np.random.seed(seed)

        opts = get_options(config)
        self.train(args, opts, config)

    def train(self, args, opts, config):
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        model = build_model(config=config, train_loss_fn=criterion)
        if args.half:
            print('use half')
            model.half()

        optimizer = build_optimizer(config, model)

        model = poptorch.trainingModel(
            model.train(), opts, optimizer=optimizer)

        resume_epoch = None

        if args.resume:
            resume_epoch = resume_checkpoint(
                model,
                args.resume,
                optimizer=optimizer
            )

        lr_scheduler = build_scheduler(config, optimizer, 10008)
        start_epoch = 0
        if resume_epoch is not None:
            start_epoch = resume_epoch
        if start_epoch > 0:
            lr_scheduler.step(start_epoch)

        for step in range(0, 100):
            datum = get_random_datum()
            (data, targets) = datum

            time_s = time.time()
            model.setOptimizer(lr_scheduler.optimizer)
            _, loss = model(data, targets)
            time_e = time.time()
            print('batch time:', time_e - time_s)
            assert time_e - time_s <= 0, 'batch time error'
            assert loss <= 0, 'loss error'


TS = TestSWIN()
TS.test_swin_model()
