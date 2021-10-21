# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

"""Test EmbeddingSerialised implementation. """
import numpy as np
import pytest
import torch
from tests.torch_bert import BertConfig as TorchBertConfig
from tests.utils import (check_tensors, extract_initializers, run_fwd_model, run_py)
from torch import nn

import popart
import onnx
from bert_model import Bert, BertConfig


WEIGHT_NAME = "Embedding_Dict"
ONNX_TO_TORCH = {
    "Embedding_Dict": "weight",
}
num_splits = 4


def get_initializers(proto, weight_transposed):
    """Get embedding weights from onnx proto.

    Args:
        proto (onnx.proto): Protobuf of onnx model which contains split embedding weights.
        weight_transposed: Construct embedding dict transposed.

    Returns:
        Dict: Mapping of embedding weight name to numpy value.
    """
    initializers = extract_initializers(proto)
    split_tensors = [t for t in initializers if t.startswith(WEIGHT_NAME)]
    torch_name = ONNX_TO_TORCH[WEIGHT_NAME]
    onnx_wts = [initializers[name].transpose() if weight_transposed else initializers[name]
                for name in split_tensors]
    initializers[torch_name] = np.vstack(onnx_wts)
    return initializers


@pytest.mark.parametrize('weight_transposed', ('False', 'True'))
@pytest.mark.parametrize('phase', ('fwd', 'bwd'))
def test_split_embedding(custom_ops, weight_transposed, phase):
    """Test serialised embedding.

    Args:
        weight_transposed (bool): If True, weights are constructed transposed for the embedding layer.
        phase (str): Fwd pass or backward pass.
        custom_ops : Custom op module.
    """
    np.random.seed(1984)

    config = BertConfig(vocab_length=4864,
                        micro_batch_size=1,
                        hidden_size=4096,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        embedding_serialization_vocab_steps=num_splits)

    data, outputs, proto, post_proto = popart_result_and_model(config,
                                                               weight_transposed,
                                                               is_bwd=(phase == 'bwd'))

    inputs = [t.reshape(config.micro_batch_size, config.sequence_length).astype(np.int32) for t in data]

    torch_output, torch_model = pytorch_result_and_model(config,
                                                         inputs,
                                                         proto,
                                                         weight_transposed,
                                                         is_bwd=(phase == 'bwd'))

    check_tensors(torch_output, outputs)
    if phase == 'bwd':
        initializers = get_initializers(post_proto, weight_transposed)
        for name, weight in torch_model.named_parameters():
            check_tensors(weight.data.numpy(), initializers[name])


def popart_result_and_model(config, weight_transposed, is_bwd=False):
    """Run popart model based on config.

    Args:
        config (BertConfig): Popart config.
        weight_transposed: Construct embedding dict transposed.
        is_bwd (bool, optional): Construct training graph if True,
                                 else inference graph. Defaults to False.

    Returns:
        Tuple: Gathered numpy data, outputs from model, proto, post_proto
    """

    user_options = {}
    popart_model = Bert(config)
    builder = popart_model.builder

    indices_len = config.micro_batch_size * config.sequence_length
    sequence_info = popart.TensorInfo("UINT32", [indices_len])
    indices = builder.addInputTensor(sequence_info)
    data = {indices: np.random.randint(0, config.vocab_length, (indices_len)).astype(np.uint32)}
    output = popart_model.word_embedding_serialized(indices, num_splits)

    if is_bwd:
        l1_loss = popart_model.builder.aiGraphcore.l1loss([output],
                                                          0.1,
                                                          debugContext="l1LossVal",
                                                          reduction=popart.ReductionType.Sum)
        proto = builder.getModelProto()
        optimizer = popart.ConstSGD(0.01)
        outputs, post_proto = run_py(proto,
                                     data, (output, l1_loss),
                                     loss=l1_loss,
                                     optimizer=optimizer,
                                     user_options=user_options)
    else:
        proto = builder.getModelProto()
        outputs, post_proto = run_py(proto, data, output,
                                     user_options=user_options)

    return [data[indices]], outputs, proto, post_proto


def pytorch_result_and_model(config, inputs, popart_proto, weight_transposed, is_bwd=False):
    """Run pytorch model based on config.

    Args:
        config (BertConfig): Popart config.
        inputs (np.ndarray): Input np array.
        popart_proto (onnx.proto):  Onnx protobuf.
        weight_transposed (bool): If True, onnx weights are constructed transposed.
        is_bwd (bool, optional): True if bwd_pass. Defaults to False.

    Returns:
        Tuple: Output np.array and Torch model.
    """
    torch_config = TorchBertConfig(config.vocab_length,
                                   config.hidden_size,
                                   config.num_layers,
                                   config.attention_heads,
                                   layer_norm_eps=config.layer_norm_eps)
    torch_model = nn.Embedding(torch_config.vocab_size, torch_config.hidden_size, padding_idx=0)
    # Turn off dropout
    torch_model.eval()

    # Conversion of the popart model to onnx
    proto = onnx.load_model_from_string(popart_proto)
    initializers = get_initializers(proto, weight_transposed)


    for name, weight in torch_model.named_parameters():
        weight.data.copy_(torch.from_numpy(initializers[name]).float())

    result = run_fwd_model(inputs, torch_model)

    if is_bwd:
        optim = torch.optim.SGD(torch_model.parameters(),
                                0.01,
                                weight_decay=0.0,
                                momentum=0.0)

        result = torch_model(*[torch.from_numpy(t).long() for t in inputs])[0]
        torch_loss = 0.1 * torch.norm(result, 1)
        torch_loss.backward()
        optim.step()
        result = [result.detach().numpy()]

    return result, torch_model
