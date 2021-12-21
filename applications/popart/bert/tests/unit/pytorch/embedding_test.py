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
import popart
import pytest
import onnx

from bert_model import Bert, BertConfig
from tests.torch_bert import BertConfig as TorchBertConfig, BertEmbeddings
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model


'''
Tests the embedding with no projection. This is the case for SQUAD.
'''

TORCH_TO_ONNX = {
    "word_embeddings.weight": "Embedding/Embedding_Dict",
    "position_embeddings.weight": "Embedding/Positional_Dict",
    "token_type_embeddings.weight": "Embedding/Segment_Dict",
    "LayerNorm.weight": "Embedding/Gamma",
    "LayerNorm.bias": "Embedding/Beta"
}


def setup_function(func):
    np.random.seed(1984)


def test_embedding_fwd(custom_ops):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        micro_batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        inference=True)
    popart_model = Bert(config)

    sequence_info = popart.TensorInfo(
        "UINT32", [config.micro_batch_size * config.sequence_length])
    indices = popart_model.builder.addInputTensor(sequence_info)
    positions = popart_model.builder.addInputTensor(sequence_info)
    segments = popart_model.builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32),
        positions:
        np.random.randint(0, config.max_positional_length,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32),
        segments:
        np.random.randint(0, 2,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32)
    }

    user_options = {"enableStochasticRounding": True}
    with popart_model.builder.nameScope("Embedding"):
        output = popart_model.embedding(indices, positions, segments)

    proto = popart_model.builder.getModelProto()
    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 user_options=user_options)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[t].reshape(config.micro_batch_size, config.sequence_length).astype(np.int32) for t in [indices, positions, segments]]

    #  ------------------- PyTorch -------------------------
    torch_model = BertEmbeddings(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps))
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto, TORCH_TO_ONNX, {})
    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs, margin=5e-7)


@pytest.mark.ipu_version("ipu2")
def test_embedding_bwd(custom_ops):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        micro_batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        update_embedding_dict=True)

    popart_model = Bert(config)
    # Prevent virtualGraph attributes being added to the ops

    sequence_info = popart.TensorInfo(
        "UINT32", [config.micro_batch_size * config.sequence_length])
    indices = popart_model.builder.addInputTensor(sequence_info)
    positions = popart_model.builder.addInputTensor(sequence_info)
    segments = popart_model.builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32),
        positions:
        np.random.randint(0, config.max_positional_length,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32),
        segments:
        np.random.randint(0, 2,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32)
    }

    optimizer = popart.ConstSGD(0.01)

    l1_lambda = 0.1
    with popart_model.builder.nameScope("Embedding"):
        output = popart_model.embedding(indices, positions, segments)
    l1 = popart_model.builder.aiGraphcore.l1loss(
        [output],
        l1_lambda,
        debugContext="l1LossVal",
        reduction=popart.ReductionType.Sum)

    num_reps = 5
    proto = popart_model.builder.getModelProto()
    outputs, post_proto = run_py(proto,
                                 data,
                                 output,
                                 ipus=1,
                                 loss=l1,
                                 num_reps=num_reps,
                                 optimizer=optimizer)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[t].reshape(config.micro_batch_size, config.sequence_length).astype(np.int32) for t in [indices, positions, segments]]

    #  ------------------- PyTorch -------------------------

    torch_model = BertEmbeddings(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        update_embedding_dict=config.update_embedding_dict))
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model, proto, TORCH_TO_ONNX, {})

    optim = torch.optim.SGD(torch_model.parameters(), 0.01)
    for _ in range(num_reps):
        torch_output = torch_model(*[torch.from_numpy(t).long() for t in inputs])
        torch_loss = l1_lambda * torch.norm(torch_output, 1)
        torch_loss.backward()
        optim.step()
        optim.zero_grad()

    torch_outputs = [torch_output.detach().numpy()]

    check_tensors(torch_outputs, outputs, margin=7e-6)

    check_model(torch_model, post_proto, TORCH_TO_ONNX, {}, margin=7e-06)
