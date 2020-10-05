# Copyright 2019 Graphcore Ltd.
import numpy as np
import torch
from torch import nn
import pytest

import popart
import onnx

from bert_model import BertConfig, Bert, ExecutionMode, get_model
from bert import set_library_seeds
from tests.torch_bert import BertConfig as TorchBertConfig, BertOutput, BertIntermediate
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model

'''
Tests the fully connected layers.
'''
test_modes = [ExecutionMode.DEFAULT, pytest.param(ExecutionMode.PHASED, marks=pytest.mark.requires_remote_buffers)]


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


@pytest.mark.parametrize('mode', test_modes)
@pytest.mark.parametrize('activation_function', ACTIVATIONS.keys())
@pytest.mark.parametrize('phase', ('fwd', 'bwd'))
def test_activation_function(mode, activation_function, phase):

    set_library_seeds(0)

    popart_act_function, pytorch_activation = ACTIVATIONS[activation_function]
    config = BertConfig(vocab_length=128,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        activation_type=str(popart_act_function))

    data, outputs, proto, post_proto = popart_result_and_model(
        config, mode, is_bwd=False if phase is 'fwd' else True)

    inputs = [
        data.reshape(config.batch_size, config.sequence_length,
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
        is_bwd=False if phase is 'fwd' else True)

    check_tensors(torch_output, outputs, margin=1e-07)

    if phase is 'bwd':
        check_model(torch_model,
                    post_proto,
                    TORCH_TO_ONNX[mode],
                    transform=TRANSPOSE_WEIGHTS,
                    margin=1e-07)


def popart_result_and_model(popart_config, mode, is_bwd=False):
    popart_model = get_model(popart_config, mode, 'feedforward')

    input_info = popart.TensorInfo(popart_config.popart_dtype, [
        popart_config.batch_size * popart_config.sequence_length,
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
            "batchSerializationFactor": 1,
            "executionPhases": popart_model.total_execution_phases
        }
        output = popart_model(input_tensor)
    else:
        user_options = {"enableStochasticRounding": True}
        output = popart_model.feed_forward(input_tensor)

    if is_bwd:
        l1_lambda = 0.1
        if mode == ExecutionMode.PHASED:
            with popart_model.scope_provider(popart_model.builder, popart_model.norm.scope):
                l1 = popart_model.builder.aiGraphcore.l1loss(
                    [output],
                    l1_lambda,
                    debugPrefix="l1LossVal",
                    reduction=popart.ReductionType.Sum)

        else:
            l1 = popart_model.builder.aiGraphcore.l1loss(
                [output],
                l1_lambda,
                debugPrefix="l1LossVal",
                reduction=popart.ReductionType.Sum)
        proto = popart_model.builder.getModelProto()
        optimizer = popart.ConstSGD(0.01)

        outputs, post_proto = run_py(proto,
                                     data, (output, l1),
                                     loss=l1,
                                     optimizer=optimizer,
                                     user_options=user_options,
                                     execution_mode=mode)
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
                             is_bwd=False):
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
                                0.01,
                                weight_decay=0.0,
                                momentum=0.0)

        result = torch_model(*[torch.from_numpy(t).float() for t in inputs])[0]
        torch_loss = l1_lambda * torch.norm(result, 1)
        torch_loss.backward()
        optim.step()
        result = result.detach().numpy()

    return result, torch_model
