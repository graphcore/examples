# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

"""
Weight decay Unit test.
Run backward pass with different weight_decay values.
Check tensors values for popart vs. pytorch simplified models.
"""

import numpy as np
import pytest
import torch
from torch import nn

import popart
import onnx

from bert_model import Bert, BertConfig
from bert_optimizer import BaseOptimizerFactory
from tests.torch_bert import BertConfig as TorchBertConfig
from tests.torch_bert import BertIntermediate, BertOutput
from tests.utils import (check_model, check_tensors, copy_weights_to_torch,
                         run_fwd_model, run_py)


class MockIteration:
    def __init__(self):
        self.learning_rate = 0


class MockArgs:
    def __init__(self, optimizer, learning_rate, weight_decay):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = 0
        self.dampening = 0
        self.velocity_scaling = 1.0
        self.loss_scaling = 1.0
        self.task = "PRETRAINING"
        self.weight_decay = weight_decay
        self.momentum_scaling = 0
        self.pipeline_momentum_scaling = 0
        self.execution_mode = "DEFAULT"
        self.squad_lr_scale = None
        self.continuous_pipeline_optimizer_scaling = False
        self.use_half_optimizer_state = False


class BertFCN(nn.Module):
    def __init__(self, config):
        super(BertFCN, self).__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, input_x):
        intermediate_output = self.intermediate(input_x)
        layer_output = self.output(intermediate_output, input_x)
        outputs = (layer_output, )
        return outputs


TORCH_TO_ONNX = {
    "intermediate.dense.weight": "1/W",
    "intermediate.dense.bias": "1/B",
    "output.dense.weight": "2/W",
    "output.dense.bias": "2/B",
    "output.LayerNorm.weight": "Gamma",
    "output.LayerNorm.bias": "Beta"
}

TRANSPOSE_WEIGHTS = {
    "intermediate.dense.weight": np.transpose,
    "output.dense.weight": np.transpose
}


@pytest.mark.parametrize('weight_decay', [0.5, 0.1])
def test_weight_decay(weight_decay):

    lr = 0.01
    l1_lambda = 0.1

    #  ------------------- PopART -------------------------
    config = BertConfig(vocab_length=128,
                        micro_batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        activation_type='Gelu')

    data, outputs, proto, post_proto = popart_result_and_model(
        config, weight_decay=weight_decay, lr=lr, l1_lambda=l1_lambda)

    #  ------------------- PyTorch -------------------------
    torch_config = TorchBertConfig(config.vocab_length,
                                   config.hidden_size,
                                   config.num_layers,
                                   config.attention_heads,
                                   layer_norm_eps=config.layer_norm_eps,
                                   hidden_dropout_prob=0.,
                                   hidden_act=nn.functional.gelu)

    inputs = [
        data.reshape(config.micro_batch_size, config.sequence_length,
                     config.hidden_size)
    ]

    torch_output, torch_model = pytorch_result_and_model(
        torch_config,
        inputs,
        proto,
        weight_decay=weight_decay,
        lr=lr,
        l1_lambda=l1_lambda)

    #  ------------------- Check outputs -------------------------
    check_tensors(torch_output, outputs)

    check_model(torch_model,
                post_proto,
                TORCH_TO_ONNX,
                transform=TRANSPOSE_WEIGHTS)


def popart_result_and_model(popart_config, weight_decay=0.0, lr=0.0, l1_lambda=0.0):
    builder = popart.Builder()
    popart_model = Bert(popart_config, builder=builder)

    input_info = popart.TensorInfo(popart_config.popart_dtype, [
        popart_config.micro_batch_size * popart_config.sequence_length,
        popart_config.hidden_size
    ])
    input_tensor = builder.addInputTensor(input_info)

    data = {
        input_tensor:
        np.random.normal(0, 0.02,
                         input_info.shape()).astype(popart_config.dtype)
    }

    output = popart_model.feed_forward(input_tensor)

    l1 = builder.aiGraphcore.l1loss([output], l1_lambda, debugContext="l1LossVal", reduction=popart.ReductionType.Sum)
    proto = builder.getModelProto()

    iteration = MockIteration()
    args = MockArgs("SGD", lr, weight_decay)
    optimizer_factory = BaseOptimizerFactory(args, iteration,
                                             popart_model.tensors)
    optimizer = optimizer_factory.create()

    outputs, post_proto = run_py(proto,
                                 data, (output, l1),
                                 loss=l1,
                                 optimizer=optimizer)

    return data[input_tensor], outputs, proto, post_proto


def pytorch_result_and_model(torch_config,
                             inputs,
                             popart_proto,
                             weight_decay=0.0,
                             lr=0.0,
                             l1_lambda=0.0):

    proto = onnx.load_model_from_string(popart_proto)
    torch_model = BertFCN(torch_config)
    torch_model.eval()  # Turn off dropout
    copy_weights_to_torch(torch_model,
                          proto,
                          TORCH_TO_ONNX,
                          transform=TRANSPOSE_WEIGHTS)
    run_fwd_model(inputs, torch_model)

    decay = []
    no_decay = []
    for name, param in torch_model.named_parameters():
        if "bias" in name or "LayerNorm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    params = [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }]

    optim = torch.optim.SGD(params, lr, momentum=0.0)

    result = torch_model(*[torch.from_numpy(t).float() for t in inputs])[0]
    torch_loss = l1_lambda * torch.norm(result, 1)
    torch_loss.backward()
    optim.step()
    result = result.detach().numpy()

    return result, torch_model
