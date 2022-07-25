# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2019 Shigeki Karita
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
'''
This script has been adapted from some of the original EspNet found here:
[
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py
]

Main changes:
    rename the parameters of forward function
'''

import torch
import poptorch as pt
from torch import nn


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self, ignore_idx=-1):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)

    def forward(self, x, target):
        loss = self.ce_loss(x.transpose(1, 2), target.long())
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing, vocab_size, normalize_length=False):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing
        self.other_value = self.smoothing / (self.vocab_size - 1)
        self.offset = self.confidence - self.other_value
        self.normalize_length = normalize_length

    def forward(self, logits, target, target_mask):
        y_true = torch.nn.functional.one_hot(target.long(), self.vocab_size).to(logits.dtype)
        y_true = y_true * self.offset + self.other_value
        y_pred = torch.log_softmax(logits, -1)
        loss_pre = y_true * (torch.log(y_true) - y_pred) * target_mask.unsqueeze(-1)
        loss = torch.sum(loss_pre)
        if self.normalize_length:
            loss = loss / target_mask.int().sum()
        else:
            loss = loss / target.size(0)
        return loss
