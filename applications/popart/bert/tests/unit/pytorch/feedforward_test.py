# Copyright 2019 Graphcore Ltd.
import os
import ctypes
import numpy as np
from pathlib import Path
import torch
from torch import nn
import pytest

import popart
import onnx

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertOutput, BertIntermediate
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model


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
        outputs = (layer_output, )  # + attention_outputs[1:]
        return outputs


ACTIVATIONS = {
    'Relu': ['Relu', nn.functional.relu],
    'Gelu': ['Gelu', nn.functional.gelu],
    'SGelu': ['SGelu', simplified_gelu],
    'Leaky_relu':
    ['LRelu', lambda x: nn.functional.leaky_relu(x, negative_slope=0.1)]
}

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


@pytest.mark.parametrize(
    ('activation_function', 'phase'),
    [(i, p) for i in ACTIVATIONS.keys() for p in ['fwd', 'bwd']])
def test_activation_function(activation_function, phase, custom_ops):
    popart_act_function, pytorch_activation = ACTIVATIONS[activation_function]
    config = BertConfig(vocab_length=128,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=[],
                        activation_type=str(popart_act_function))

    data, outputs, proto, post_proto = popart_result_and_model(
        config, is_bwd=False if phase is 'fwd' else True)

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
        torch_config, inputs, proto, is_bwd=False if phase is 'fwd' else True)

    check_tensors(torch_output, outputs)

    if phase is 'bwd':
        check_model(torch_model,
                    post_proto,
                    TORCH_TO_ONNX,
                    transform=TRANSPOSE_WEIGHTS)


def popart_result_and_model(popart_config, is_bwd=False):
    builder = popart.Builder()
    popart_model = Bert(popart_config, builder=builder)

    input_info = popart.TensorInfo(popart_config.popart_dtype, [
        popart_config.batch_size * popart_config.sequence_length,
        popart_config.hidden_size
    ])
    input_tensor = builder.addInputTensor(input_info)

    data = {
        input_tensor:
        np.random.normal(0, 0.02,
                         input_info.shape()).astype(popart_config.dtype)
    }

    output = popart_model.feed_forward(input_tensor)
    proto = builder.getModelProto()

    if is_bwd:
        l1_lambda = 0.1
        l1 = popart.L1Loss(output, "l1LossVal", l1_lambda)
        optimizer = popart.ConstSGD(0.01)

        outputs, post_proto = run_py(proto,
                                     data, (output, l1.output(0)),
                                     loss=l1,
                                     optimizer=optimizer)
    else:
        outputs, post_proto = run_py(proto, data, output)

    return data[input_tensor], outputs, proto, post_proto


def pytorch_result_and_model(torch_config, inputs, popart_proto, is_bwd=False):
    # Conversion of the popart model to onnx
    proto = onnx.load_model_from_string(popart_proto)

    torch_model = BertFCN(torch_config)
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model,
                          proto,
                          TORCH_TO_ONNX,
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
