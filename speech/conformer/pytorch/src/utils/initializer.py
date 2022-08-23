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
    https://github.com/espnet/espnet/blob/master/espnet2/torch_utils/initialize.py
]
Main change:
    remove the LegacyRelPositionMultiHeadedAttention class.

'''

import torch
from src.layers.layer_norm import LayerNorm


def initialize(model, init_type="pytorch"):
    """Initialize Transformer module.

    :param torch.nn.Module model: transformer instance
    :param str init_type: initialization type
    """
    if init_type == "pytorch":
        return

    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init_type)
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()

    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, LayerNorm)):
            m.reset_parameters()
