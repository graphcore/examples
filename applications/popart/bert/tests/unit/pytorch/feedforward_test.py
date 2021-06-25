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

import numpy as np
import torch
from torch import nn
import pytest

import popart
import onnx

from bert_model import BertConfig, Bert, ExecutionMode, get_model
from bert import set_library_seeds
from tests.torch_bert import BertConfig as TorchBertConfig, BertOutput, BertIntermediate
from tests.utils import (
    run_py,
    copy_weights_to_torch,
    run_fwd_model,
    check_tensors,
    check_model,
    requires_remote_buffers,
    sanity)

'''
Tests the fully connected layers.
'''
test_modes = [ExecutionMode.DEFAULT, requires_remote_buffers(ExecutionMode.PHASED)]
num_reps_bwd = 5
lr = 1e-3


def simplified_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class BertFCN(nn.Module):
    def __init__(self, config):
        super(BertFCN, self).__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, input_x):
        intermediate_output = self.intermediate(input_x)
        layer_output = self.output(intermediate_output, input_x)
        # add attentions if we output them
        outputs = (layer_output, )
        return outputs


ACTIVATIONS = {
    'Relu': ['Relu', nn.functional.relu],
    'Gelu': ['Gelu', nn.functional.gelu],
    'SGelu': ['SGelu', simplified_gelu],
    'Leaky_relu':
    ['LRelu', lambda x: nn.functional.leaky_relu(x, negative_slope=0.1)]
}

TORCH_TO_ONNX = {
    ExecutionMode.DEFAULT: {
        "intermediate.dense.weight": "1/W",
        "intermediate.dense.bias": "1/B",
        "output.dense.weight": "2/W",
        "output.dense.bias": "2/B",
        "output.LayerNorm.weight": "Gamma",
        "output.LayerNorm.bias": "Beta"
    },
    ExecutionMode.PHASED: {
        "intermediate.dense.weight": "FF/1/Dense/Weight",
        "intermediate.dense.bias": "FF/1/Dense/Bias",
        "output.dense.weight": "FF/2/Dense/Weight",
        "output.dense.bias": "FF/2/Dense/Bias",
        "output.LayerNorm.weight": "FF/Norm/Gamma",
        "output.LayerNorm.bias": "FF/Norm/Beta"
    }
}

TRANSPOSE_WEIGHTS = {
    "intermediate.dense.weight": np.transpose,
    "output.dense.weight": np.transpose
}


def setup_function(func):
    np.random.seed(1984)


@pytest.mark.parametrize('mode', test_modes)
@pytest.mark.parametrize('micro_batch_size, batch_serialization_factor, phase, momentum',
                         [sanity(4, 2, "bwd", 0.984375),

                          (1, 1, "fwd", 0.0),
                          (2, 1, "fwd", 0.0),
                          (2, 2, "fwd", 0.0),
                          (4, 2, "fwd", 0.0),
                          (1, 1, "bwd", 0.0),
                          (2, 1, "bwd", 0.0),
                          (2, 2, "bwd", 0.0),
                          (4, 2, "bwd", 0.0),
                          (1, 1, "bwd", 0.984375),
                          (2, 1, "bwd", 0.984375),
                          (2, 2, "bwd", 0.984375),
                          (4, 2, "bwd", 0.984375)])
def test_activation_function(mode, phase, momentum, micro_batch_size, batch_serialization_factor):

    set_library_seeds(0)

    popart_act_function, pytorch_activation = ACTIVATIONS["Gelu"]
    config = BertConfig(vocab_length=128,
                        micro_batch_size=micro_batch_size,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        activation_type=str(popart_act_function))

    data, outputs, proto, post_proto = popart_result_and_model(
        config, mode, batch_serialization_factor, is_bwd=False if phase is 'fwd' else True, momentum=momentum)

    inputs = [
        data.reshape(config.micro_batch_size, config.sequence_length,
                     config.hidden_size)
    ]

    #  ------------------- PyTorch -------------------------
    torch_config = TorchBertConfig(config.vocab_length,
                                   config.hidden_size,
                                   config.num_layers,
                                   config.attention_heads,
                                   layer_norm_eps=config.layer_norm_eps,
                                   hidden_dropout_prob=0.,
                                   hidden_act=pytorch_activation)

    torch_output, torch_model = pytorch_result_and_model(
        torch_config,
        inputs,
        proto,
        mode,
        is_bwd=False if phase is 'fwd' else True,
        momentum=momentum)

    check_tensors(torch_output, outputs, margin=7e-6)

    if phase is 'bwd':
        check_model(torch_model,
                    post_proto,
                    TORCH_TO_ONNX[mode],
                    transform=TRANSPOSE_WEIGHTS,
                    margin=7e-6)


def popart_result_and_model(popart_config, mode, batch_serialization_factor, is_bwd=False, momentum=0.0):
    popart_model = get_model(popart_config, mode, 'feedforward')

    input_info = popart.TensorInfo(popart_config.popart_dtype, [
        popart_config.micro_batch_size * popart_config.sequence_length,
        popart_config.hidden_size
    ])
    input_tensor = popart_model.builder.addInputTensor(input_info)

    data = {
        input_tensor:
        np.random.normal(0, 0.02,
                         input_info.shape()).astype(popart_config.dtype)
    }

    user_options = {}
    if mode == ExecutionMode.PHASED:
        user_options = {
            "batchSerializationFactor": batch_serialization_factor,
            "executionPhases": popart_model.total_execution_phases,
        }
        output = popart_model(input_tensor)
    else:
        output = popart_model.feed_forward(input_tensor)

    if is_bwd:
        l1_lambda = 0.1
        if mode == ExecutionMode.PHASED:
            with popart_model.scope_provider(popart_model.builder, popart_model.norm.scope):
                l1 = popart_model.builder.aiGraphcore.l1loss(
                    [output],
                    l1_lambda,
                    debugContext="l1LossVal",
                    reduction=popart.ReductionType.Sum)

        else:
            l1 = popart_model.builder.aiGraphcore.l1loss(
                [output],
                l1_lambda,
                debugContext="l1LossVal",
                reduction=popart.ReductionType.Sum)
        proto = popart_model.builder.getModelProto()

        if momentum > 0.0:
            optimizer = popart.SGD({"defaultLearningRate": (lr, False),
                                    "defaultMomentum": (momentum, False),
                                    "defaultWeightDecay": (0.0, False)})
        else:
            optimizer = popart.ConstSGD(lr)

        outputs, post_proto = run_py(proto,
                                     data, (output, l1),
                                     loss=l1,
                                     optimizer=optimizer,
                                     user_options=user_options,
                                     execution_mode=mode,
                                     num_reps=num_reps_bwd)
    else:
        proto = popart_model.builder.getModelProto()
        outputs, post_proto = run_py(proto,
                                     data,
                                     output,
                                     user_options=user_options,
                                     execution_mode=mode)

    return data[input_tensor], outputs, proto, post_proto


def pytorch_result_and_model(torch_config,
                             inputs,
                             popart_proto,
                             mode,
                             is_bwd=False,
                             momentum=0.0):
    # Conversion of the popart model to onnx
    proto = onnx.load_model_from_string(popart_proto)

    torch_model = BertFCN(torch_config)
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model,
                          proto,
                          TORCH_TO_ONNX[mode],
                          transform=TRANSPOSE_WEIGHTS)

    result = run_fwd_model(inputs, torch_model)

    if is_bwd:
        l1_lambda = 0.1
        optim = torch.optim.SGD(torch_model.parameters(),
                                lr,
                                weight_decay=0.0,
                                momentum=momentum)

        if momentum > 0.0:
            for group in optim.param_groups:
                for p in group['params']:
                    optim.state[p]['momentum_buffer'] = p.data * 0.
                    optim.state[p]['exp_avg'] = p.data * 0.
                    optim.state[p]['exp_avg_sq'] = p.data * 0.
                    optim.state[p]['step'] = 0


        for _ in range(num_reps_bwd):
            result = torch_model(*[torch.from_numpy(t).float() for t in inputs])[0]
            torch_loss = l1_lambda * torch.norm(result, 1)
            torch_loss.backward()
            optim.step()
            optim.zero_grad()
        result = [result.detach().numpy()]

    return result, torch_model
