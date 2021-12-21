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
import numpy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import poptorch
import popart
from pathlib import Path
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling as Conv2dSubsampling_cpu,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    RelPositionalEncoding as RelPositionalEncoding_cpu,
)
from espnet.nets.pytorch_backend.conformer.swish import Swish as Swish_cpu
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm as LayerNorm_cpu
from src.layers.embedding import RelPositionalEncoding
from src.layers.subsampling import Conv2dSubsampling
from src.utils.initializer import initialize
from src.layers.swish import Swish
from src.layers.layer_norm import LayerNorm
from src.layers.convolution import ConvolutionModule
from tests.convolution import ConvolutionModule_cpu


def parse_conformer_args():
    yaml_args = dict()
    app_path = str(Path(__file__).parent.parent.resolve())
    config_name = Path(app_path, 'configs', 'train.yaml')
    if config_name is not None:
        with open(config_name, 'r') as f:
            try:
                yaml_args = yaml.safe_load(f)
                return yaml_args
            except yaml.YAMLError as exc:
                sys.exit(1)


@pytest.mark.ipus(3)
class ConvGradIpuModel(nn.Module):
    def __init__(self):
        super(ConvGradIpuModel, self).__init__()
        yaml_args = parse_conformer_args()
        activation = Swish()
        output_size = yaml_args['encoder']['output_size']
        dropout_rate = 0
        convolution_layer = ConvolutionModule
        cnn_module_kernel = yaml_args['encoder']['cnn_module_kernel']
        convolution_layer_args = (output_size, cnn_module_kernel, activation)
        self.conv_module = convolution_layer(*convolution_layer_args, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = nn.NLLLoss()

    def forward(self, x, target):
        stoch_layer_coeff = 1.0
        residual = x
        x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
        targes = torch.ones(24).long()
        return x, self.loss(x[0], targes)


class ConvGradCpuModel(nn.Module):
    def __init__(self):
        super(ConvGradCpuModel, self).__init__()

        activation = Swish_cpu()
        yaml_args = parse_conformer_args()
        output_size = yaml_args['encoder']['output_size']
        kernel_zie = yaml_args['encoder']['cnn_module_kernel']
        dropout_rate = 0
        convolution_layer = ConvolutionModule_cpu
        convolution_layer_args = (output_size, kernel_zie, activation)
        self.conv_module = convolution_layer(*convolution_layer_args, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = nn.NLLLoss()

    def forward(self, x, target):
        stoch_layer_coeff = 1.0
        residual = x
        x = residual + stoch_layer_coeff * \
            self.dropout(self.conv_module(x))
        targes = torch.ones(24).long()

        return x, self.loss(x[0], targes)


class LnGradIpuModel(nn.Module):
    def __init__(self):
        super(LnGradIpuModel, self).__init__()

        yaml_args = parse_conformer_args()
        output_size = yaml_args['encoder']['output_size']

        self.norm_ff = LayerNorm(output_size, eps_=1e-12)

        self.loss = nn.NLLLoss()

    def forward(self, x, target):

        x = self.norm_ff(x)
        targes = torch.ones(24).long()
        return x, self.loss(x[0], targes)


class LnGradCpuModel(nn.Module):
    def __init__(self):
        super(LnGradCpuModel, self).__init__()
        yaml_args = parse_conformer_args()

        output_size = yaml_args['encoder']['output_size']
        self.norm_ff = LayerNorm_cpu(output_size)

        self.loss = nn.NLLLoss()

    def forward(self, x, target):
        x = self.norm_ff(x)

        targes = torch.ones(24).long()

        return x, self.loss(x[0], targes)


class SubGradIpuModel(nn.Module):
    def __init__(self):
        super(SubGradIpuModel, self).__init__()

        pos_enc_class = RelPositionalEncoding
        yaml_args = parse_conformer_args()
        output_size = yaml_args['encoder']['output_size']
        feature_size = yaml_args['encoder']['input_size']
        self.embed = Conv2dSubsampling(
            feature_size, 32, 0, pos_enc_class(output_size, 0, output_size)
        )
        self.loss = nn.NLLLoss()

    def forward(self, xs_pad, masks):

        xs_pad, masks = self.embed(xs_pad, masks)
        targes = torch.ones(1, 32).long()

        return xs_pad, self.loss(xs_pad[0], targes)


class SubGradCpuModel(nn.Module):
    def __init__(self):
        super(SubGradCpuModel, self).__init__()

        pos_enc_class = RelPositionalEncoding_cpu
        yaml_args = parse_conformer_args()
        output_size = yaml_args['encoder']['output_size']
        feature_size = yaml_args['encoder']['input_size']
        self.embed = Conv2dSubsampling_cpu(
            feature_size, 32, 0, pos_enc_class(output_size, 0, output_size)
        )
        self.loss = nn.NLLLoss()

    def forward(self, xs_pad, masks):

        xs_pad, masks = self.embed(xs_pad, masks)
        targes = torch.ones(1, 32).long()

        return xs_pad, self.loss(xs_pad[0], targes)


def get_dict():

    dict_shape = {}
    yaml_args = parse_conformer_args()
    output_size = yaml_args['encoder']['output_size']
    feature_size = yaml_args['encoder']['input_size']

    dict_shape['sub'] = [[1, 100, feature_size], [1, 1, 100]]
    dict_shape['conv'] = [[32, 24, output_size], [32, 1, 24]]
    dict_shape['ln'] = [[32, 24, output_size], [32, 1, 24]]
    return dict_shape
