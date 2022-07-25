# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2017-present, Facebook, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import poptorch
import torch

# Code below derived from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py


def mixup_data(x, y, alpha=1.0, seed=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if seed is not None:
        np.random.seed(seed)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, torch.stack([torch.tensor(lam)]*batch_size)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss1 = lam * criterion(pred, y_a)
    loss2 = (1 - lam) * criterion(pred, y_b)
    return poptorch.identity_loss(loss1 + loss2, reduction='none')
