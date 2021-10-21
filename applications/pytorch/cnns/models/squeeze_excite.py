# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Hacked together by / Copyright 2020 Ross Wightman
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
import torch
from torch.nn import functional as F


class SqueezeExciteIPU(torch.nn.Module):
    """
    Squeeze-Excitation layer that is the same as the timm SE layer but uses
    average pooling (instead of mean reduction) for the 'squeeze' part.
    """
    def __init__(self, se_timm_layer):
        super(SqueezeExciteIPU, self).__init__()
        self.conv_reduce = se_timm_layer.conv_reduce
        self.act1 = se_timm_layer.act1
        self.conv_expand = se_timm_layer.conv_expand
        self.gate = se_timm_layer.gate

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
