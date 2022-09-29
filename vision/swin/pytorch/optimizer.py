# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Swin Transformer
# This file has been modified by Graphcore Ltd.
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# The LICENSE referenced above is reproduced below:
# MIT License
#
#     Copyright (c) Microsoft Corporation.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE
# Written by Ze Liu
# --------------------------------------------------------
import popdist
import torch
from torch import optim as optim
from poptorch.optim import AdamW, SGD
import horovod.torch as hvd


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = SGD(parameters,
                        lr=config.TRAIN.BASE_LR,
                        momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                        weight_decay=config.TRAIN.WEIGHT_DECAY,
                        loss_scaling=config.TRAIN.LOSS_SCALING,
                        accum_type=torch.float16,
                        use_combined_accum=True)

    elif opt_lower == 'adamw':
        if config.PRECISION[0] == 'float':
            accum_type = torch.float32
        else:
            accum_type = torch.float16
        optimizer = AdamW(parameters,
                          lr=config.TRAIN.BASE_LR,
                          betas=config.TRAIN.OPTIMIZER.BETAS,
                          eps=config.TRAIN.OPTIMIZER.EPS,
                          weight_decay=config.TRAIN.WEIGHT_DECAY,
                          loss_scaling=config.TRAIN.LOSS_SCALING,
                          accum_type=accum_type,
                          first_order_momentum_accum_type=torch.float16,
                          second_order_momentum_accum_type=torch.float32,
                          max_grad_norm=config.TRAIN.CLIP_GRAD)
    if popdist.isPopdistEnvSet():
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(
                param.shape) == 1 or name.endswith(".bias") or (
                name in skip_list) or check_keywords_in_name(
                name,
                skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
