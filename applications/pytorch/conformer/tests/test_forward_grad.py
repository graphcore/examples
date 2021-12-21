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

import time
import pytest
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import popart
import poptorch
from tests.modeling import (
    ConvGradIpuModel,
    ConvGradCpuModel,
    LnGradIpuModel,
    LnGradCpuModel,
    SubGradIpuModel,
    SubGradCpuModel,
    get_dict,
)


bs = 16
device_iterations = 1
num_workers = 1


class CompareFwdBwdPass():
    def __init__(self):
        self.model = None
        self.input_shape = None
        self.mask_shape = None
        self.isCpuOrIpu = 0
        self.masks = None
        self.xs_pad = None
        self.unit_layer = 'sub'
        self.opts = poptorch.Options()
        np.random.seed(42)
        self.opts.randomSeed(42)
        self.build_logger()

    def build_logger(self):
        logger_args = {}
        logger_args['name'] = 'conformer_logger'
        logger_args['log_file'] = 'log_test.txt'
        logger_args['level'] = 'info'

        log_level = {'info': logging.INFO, 'debug': logging.DEBUG}.get(
            logger_args['level'], logging.INFO
        )
        logger = logging.getLogger(logger_args['name'])
        logger.setLevel(level=log_level)
        formatter = logging.Formatter(
            '%(asctime)s %(filename)s line:%(lineno)d %(levelname)s %(message)s'
        )
        handler = logging.FileHandler(logger_args['log_file'])
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
        self.logger = logger

    def get_model_conf(self):
        if self.unit_layer == 'conv':
            self.model = [ConvGradIpuModel(), ConvGradCpuModel()]
            dict_shape = get_dict()
            self.input_shape = dict_shape['conv'][0]
            self.mask_shape = dict_shape['conv'][1]
            self.tensor_ = [
                'conv_module.pointwise_conv1.weight',
                'conv_module.depthwise_conv.weight',
                'conv_module.norm.weight',
                'conv_module.pointwise_conv2.weight',
                'conv_module.norm.bias'
                ]

        elif self.unit_layer == 'sub':
            self.model = [SubGradIpuModel(), SubGradCpuModel()]
            dict_shape = get_dict()
            self.input_shape = dict_shape['sub'][0]
            self.mask_shape = dict_shape['sub'][1]
            self.tensor_ = ['embed.conv.0.weight', 'embed.conv.0.bias', 'embed.conv.2.weight',
                            'embed.conv.2.bias', 'embed.out.0.weight', 'embed.out.0.bias']

        elif self.unit_layer == 'ln':
            self.model = [LnGradIpuModel(), LnGradCpuModel()]
            dict_shape = get_dict()
            self.input_shape = dict_shape['ln'][0]
            self.mask_shape = dict_shape['ln'][1]
            self.tensor_ = ['norm_ff.weight', 'norm_ff.bias']

    def build_model(self):
        self.get_xs_mask()
        model_ipu = self.model[0]
        model_cpu = self.model[1]
        tensor_ = self.tensor_
        tensors = []
        for t in tensor_:
            self.opts.anchorTensor('model.'+t, 'model.'+t)
            self.opts.anchorTensor('Gradient___model.'+t, 'Gradient___model.'+t)
        optimizer = poptorch.optim.SGD(model_ipu.parameters(), lr=0.001)
        training_model = poptorch.trainingModel(
            model_ipu, options=self.opts, optimizer=optimizer)
        model_dict = model_cpu.state_dict()
        model_one_iter = training_model.model.state_dict()
        pretrained_dict_model_one_iter = {
            k: v for k, v in model_one_iter.items() if k in model_dict
        }
        model_dict.update(pretrained_dict_model_one_iter)
        model_cpu.load_state_dict(model_dict)
        self.model_cpu = model_cpu
        self.training_model = training_model

    def get_xs_mask(self):
        np.random.seed(1)
        L1 = np.random.randn(*self.input_shape)
        xs_pad = torch.from_numpy(L1)
        np.random.seed(1)
        L2 = np.random.randn(*self.mask_shape)
        masks = torch.from_numpy(L2)
        xs_pad = xs_pad.to(torch.float32)
        self.xs_pad = xs_pad
        self.masks = masks

    def train(self):
        self.build_model()
        self.training_model.compile(self.xs_pad, self.masks)
        output, loss = self.training_model(self.xs_pad, self.masks)
        grad_dic = {}
        tensors = []
        weight_dic = {}
        tensor_names = self.training_model.getTensorNames()
        for t in self.tensor_:
            grad_name = 'Gradient___model.'+t
            wname = "model."+t
            grad_dic[grad_name] = self.training_model.getAnchoredTensor(grad_name)
            weight_dic[wname] = self.training_model.getAnchoredTensor(wname)
        return weight_dic, grad_dic, output

    def train_cpu(self):
        self.get_xs_mask()
        model_cpu = self.model_cpu
        self.optimizer_cpu = poptorch.optim.SGD(
            model_cpu.parameters(), lr=0.001)
        self.optimizer_cpu.zero_grad()
        output, loss = model_cpu(self.xs_pad, self.masks)
        name_nums = len([(name, grad)
                         for name, grad in model_cpu.named_parameters()])

        loss = loss.requires_grad_()
        loss.backward()
        self.optimizer_cpu.step()
        para_dict = [(name, param)
                     for name, param in model_cpu.named_parameters()]
        grad_dict = {}
        weight_dict = {}
        for i in range(0, name_nums):
            grad_cpu = para_dict[i][-1].grad
            name = para_dict[i][0]
            grad_dict['Gradient___model.'+name] = grad_cpu
            weight_dict[name] = model_cpu.state_dict()[name]
        return weight_dict, grad_dict, output

    def compare(self):

        self.get_model_conf()

        tensor_w_ipu, tensor_grad_ipu, output_ = self.train()
        tensor_w_cpu, tensor_grad_cpu, output_cpu = self.train_cpu()

        for k in tensor_grad_cpu:
            self.logger.info(f'grad:{ k, torch.abs(tensor_grad_ipu[k] - tensor_grad_cpu[k]).mean()}')

            assert np.allclose(
                tensor_grad_ipu[k].numpy(), tensor_grad_cpu[k].numpy(), atol=1e-5
            )

        if len(output_) == 2 and len(output_cpu) == 2:
            assert np.allclose(
                output_[0].detach().numpy(), output_cpu[0].detach().numpy(), atol=1e-5
            )
        else:
            self.logger.info(f'key:{torch.abs(output_-output_cpu).mean(), output_.shape, output_cpu.shape}')
            assert np.allclose(
                output_.detach().numpy(), output_cpu.detach().numpy(), atol=1e-5
            )

        for k in tensor_grad_cpu:
            self.logger.info(f'grad:{ k, torch.abs(tensor_grad_ipu[k] - tensor_grad_cpu[k]).mean()}')

            assert np.allclose(
                tensor_grad_ipu[k].numpy(), tensor_grad_cpu[k].numpy(), atol=1e-5
            )


def test_grad_sub():
    test_fg = CompareFwdBwdPass()
    test_fg.unit_layer = 'sub'
    test_fg.compare()


def test_grad_conv():
    test_fg = CompareFwdBwdPass()
    test_fg.unit_layer = 'conv'
    test_fg.compare()


def test_grad_ln():
    test_fg = CompareFwdBwdPass()
    test_fg.unit_layer = 'ln'
    test_fg.compare()

if __name__ == '__main__':
    app_path = str(Path(__file__).parent.parent.resolve())
    config_name = str(Path(app_path, 'tests', 'test_forward_grad.py'))
    pytest.main(['-s', config_name])
