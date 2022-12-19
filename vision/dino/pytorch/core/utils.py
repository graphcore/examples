# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2022 Facebook, Inc. and its affiliates.
#
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
#
# This file has been modified by Graphcore Ltd.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from enum import Enum, unique
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import poptorch

import popdist
import popdist.poptorch
import horovod.torch as hvd


@unique
class Precision(Enum):
    # AMP operation with FP32 input multiplicands as well as Fp32 partial sums
    # of products
    FP32 = 'float32'
    # AMP operation with FP16 input multiplicands and FP16 partial sums of
    # products
    FP16 = 'float16'
    # AMP operation with FP16 input multiplicands and FP32 partial sums of
    # products
    MasterWeight = 'masterweight'

    def __str__(self):
        return str(self.value)


def init_popdist(args):
    if popdist.isPopdistEnvSet():
        popdist.init()
        hvd.init()
        args.use_popdist = True
        if popdist.getNumTotalReplicas() != args.replica:
            print(f"The number of replicas is overridden by PopRun."
                  f"The new value is {popdist.getNumLocalReplicas()}")
        args.replica = int(popdist.getNumLocalReplicas())
        args.popdist_rank = popdist.getInstanceIndex()
        args.popdist_size = popdist.getNumInstances()
        hvd.broadcast(torch.Tensor([args.seed]), root_rank=0)
    else:
        args.use_popdist = False


def sync_metrics(outputs, average=True):
    if popdist.isPopdistEnvSet():
        return float(hvd.allreduce(torch.Tensor([outputs]), average=average))
    else:
        return outputs


def cosine_scheduler(
        base_value,
        final_value,
        epochs,
        niter_per_ep,
        warmup_epochs=0,
        start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * \
        (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '} avg:({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    weight_g = []
    weight_v = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            if 'weight_g' in name:
                weight_g.append(param)
            elif 'weight_v' in name:
                weight_v.append(param)
            else:
                regularized.append(param)

    return [{'params': regularized, 'weight_decay': 0.04},
            {'params': not_regularized, 'weight_decay': 0.},
            {'params': weight_g, 'lr': 0., 'weight_decay': 0.04},
            {'params': weight_v, 'lr': 0., 'weight_decay': 0.04}]


def save_checkpoint(epoch, model, optimizer, center, path, mid_save=False):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'center': center,
                  'epoch': epoch}
    if mid_save:
        torch.save(save_state, f'{path}/model_{epoch}.pth')
    torch.save(save_state, f'{path}/checkpoint.pth')


def load_checkpoint(model, optimizer, path):
    assert os.path.exists(path), f'{path} not exists'
    model_state = torch.load(path)
    epoch = model_state['epoch']
    weights = model_state['model']
    optimizer_weights = model_state['optimizer']
    center = model_state['center']

    model.load_state_dict(weights)
    optimizer.load_state_dict(optimizer_weights)
    return epoch, center
