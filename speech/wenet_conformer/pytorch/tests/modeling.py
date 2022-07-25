# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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


import argparse
import time
import pytest
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from src.layers.embedding import RelPositionalEncoding
from src.layers.subsampling import Conv2dSubsampling
from src.utils.initializer import initialize
from src.layers.swish import Swish
from src.layers.layer_norm import LayerNorm
from src.layers.convolution import ConvolutionModule



def parse_conformer_args():
    yaml_args = dict()
    app_path = str(Path(__file__).parent.parent.resolve())
    config_name = Path(app_path, 'configs', 'train.yaml')
    if config_name is not None:
        with open(config_name, 'r') as f:
            try:
                yaml_args = yaml.safe_load(f)
                return yaml_args['encoder']
            except yaml.YAMLError as exc:
                sys.exit(1)


@pytest.mark.ipus(3)
class TestConvModel(nn.Module):
    def __init__(self):
        super(TestConvModel, self).__init__()
        yaml_args = parse_conformer_args()
        dropout_rate = 0
        convolution_layer_args = (yaml_args['output_size'], yaml_args['cnn_module_kernel'], Swish())
        self.conv_module = ConvolutionModule(*convolution_layer_args, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = nn.NLLLoss()

    def forward(self, x, target):
        stoch_layer_coeff = 1.0
        residual = x
        x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
        targets = torch.ones(24).long()
        return x, self.loss(x[0], targets)


class TestLnModel(nn.Module):
    def __init__(self):
        super(TestLnModel, self).__init__()

        yaml_args = parse_conformer_args()
        output_size = yaml_args['output_size']

        self.norm_ff = LayerNorm(output_size, eps_=1e-12)

        self.loss = nn.NLLLoss()

    def forward(self, x, target):

        x = self.norm_ff(x)
        targets = torch.ones(24).long()
        return x, self.loss(x[0], targets)


class TestSubsampleModel(nn.Module):
    def __init__(self, odim=32, dropout=0):
        super(TestSubsampleModel, self).__init__()

        yaml_args = parse_conformer_args()
        self.embed = Conv2dSubsampling(
            yaml_args['input_size'], odim, dropout, RelPositionalEncoding(yaml_args['output_size'], 0, yaml_args['output_size'])
        )
        self.odim = odim
        self.loss = nn.NLLLoss()

    def forward(self, xs_pad, masks):

        xs_pad, masks = self.embed(xs_pad, masks)
        targets = torch.ones(1, self.odim).long()

        return xs_pad, self.loss(xs_pad[0], targets)


def get_dict(odim=32, mask_shape_sub=100, mask_shape=24):

    dict_shape = {}
    yaml_args = parse_conformer_args()
    output_size = yaml_args['output_size']
    feature_size = yaml_args['input_size']

    dict_shape['sub'] = [[1, mask_shape_sub, feature_size], [1, 1, mask_shape_sub]]
    dict_shape['conv'] = [[odim, mask_shape, output_size], [odim, 1, mask_shape]]
    dict_shape['ln'] = [[odim, mask_shape, output_size], [odim, 1, mask_shape]]
    return dict_shape
