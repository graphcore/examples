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

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertEmbeddings, BertLMPredictionHead, BertLayerNorm
from tests.utils import run_py, copy_weights_to_torch, run_fwd_model, check_tensors, check_model, TestFailureError


'''
Tests the embedding with a projection. This is the case for PRETRAINING.
'''
TORCH_TO_ONNX = {
    "embeddings.word_embeddings.weight": "Embedding_Dict",
    "embeddings.LayerNorm.weight": "Gamma",
    "embeddings.LayerNorm.bias": "Beta",
    "cls.transform.dense.weight": "CLS/LMPredictionW",
    "cls.transform.dense.bias": "CLS/LMPredictionB",
    "cls.transform.LayerNorm.weight": "CLS/Gamma",
    "cls.transform.LayerNorm.bias": "CLS/Beta"
}
TRANSPOSE_WEIGHTS = {
    "embeddings.word_embeddings.weight": np.transpose,
    "cls.transform.dense.weight": np.transpose
}


class BertEmbeddingsNoPosition(nn.Module):
    def __init__(self, config):
        super(BertEmbeddingsNoPosition, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = words_embeddings.detach()
        embeddings = self.LayerNorm(words_embeddings)
        return embeddings


class EmbeddingProjectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddingsNoPosition(config)
        self.cls = BertLMPredictionHead(config)
        self.tie_weights()

    def tie_weights(self):
        self.cls.decoder.weight = self.embeddings.word_embeddings.weight
        self.embeddings.word_embeddings.weight.data = \
            self.cls.decoder.weight.data.detach()

    def forward(self, indices):
        sequence_output = self.embeddings(indices)
        projected = self.cls(sequence_output)
        return projected


def test_embedding_projection_fwd(custom_ops):
    #  ------------------- PopART --------------------
    config = BertConfig(vocab_length=9728,
                        embedding_serialization_vocab_steps=4,
                        micro_batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_cls_layer=False,
                        inference=True)
    popart_model = Bert(config)
    builder = popart_model.builder

    sequence_info = popart.TensorInfo(
        "UINT32", [config.micro_batch_size * config.sequence_length])
    indices = builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32)
    }

    x = popart_model.gather(
        indices, config.vocab_length, "Embedding_Dict")
    x = popart_model.norm(x)
    x = popart_model.dropout(x)
    with popart_model.builder.nameScope("CLS"):
        x = popart_model.lm_prediction_head(x)
    output = popart_model.projection(x)

    proto = builder.getModelProto()

    outputs, post_proto = run_py(proto, data, output)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[indices].reshape(
        config.micro_batch_size, config.sequence_length).astype(np.int32)]

    #  ------------------- PyTorch -------------------------
    torch_model = EmbeddingProjectionModel(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        no_cls_layer=config.no_cls_layer))

    torch_model.eval()

    copy_weights_to_torch(torch_model, proto, TORCH_TO_ONNX,
                          TRANSPOSE_WEIGHTS)
    torch_model.tie_weights()

    torch_outputs = run_fwd_model(inputs, torch_model)

    check_tensors(torch_outputs, outputs)


def test_embedding_projection_bwd(custom_ops):
    l1_lambda = 0.1

    #  ------------------- PopART --------------------
    config = BertConfig(vocab_length=9728,
                        embedding_serialization_vocab_steps=4,
                        micro_batch_size=1,
                        hidden_size=288,
                        sequence_length=128,
                        activation_type='relu',
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_cls_layer=False,
                        # Currently updating embedding dict with projection is only
                        # available with momentum. And PopART != Pytorch momentum
                        # due to a bootstrapping step on iter 0.
                        update_embedding_dict=False)
    popart_model = Bert(config)
    builder = popart_model.builder

    sequence_info = popart.TensorInfo(
        "UINT32", [config.micro_batch_size * config.sequence_length])
    indices = builder.addInputTensor(sequence_info)
    data = {
        indices:
        np.random.randint(0, config.vocab_length,
                          (config.micro_batch_size * config.sequence_length)).astype(
                              np.uint32)
    }

    x = popart_model.gather(
        indices, config.vocab_length, "Embedding_Dict")
    x = popart_model.norm(x)
    x = popart_model.dropout(x)
    with popart_model.device_scope(nameScope="CLS"):
        x = popart_model.lm_prediction_head(x)
    output = popart_model.projection(x)
    l1 = builder.aiGraphcore.l1loss(
        [output], l1_lambda, debugContext="l1LossVal", reduction=popart.ReductionType.Sum)

    proto = builder.getModelProto()

    optimizer = popart.ConstSGD(0.01)

    outputs, post_proto = run_py(proto,
                                 data, output,
                                 loss=l1,
                                 optimizer=optimizer)

    # ----------------- PopART -> PyTorch ----------------
    proto = onnx.load_model_from_string(proto)

    inputs = [data[indices].reshape(
        config.micro_batch_size, config.sequence_length).astype(np.int32)]

    #  ------------------- PyTorch -------------------------

    torch_model = EmbeddingProjectionModel(
        TorchBertConfig(config.vocab_length,
                        config.hidden_size,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        no_cls_layer=config.no_cls_layer,
                        update_embedding_dict=config.update_embedding_dict))
    # Turn off dropout
    torch_model.eval()

    copy_weights_to_torch(torch_model,
                          proto,
                          TORCH_TO_ONNX,
                          transform=TRANSPOSE_WEIGHTS)

    optim = torch.optim.SGD(torch_model.parameters(),
                            0.01,
                            weight_decay=0.0,
                            momentum=0.0)

    torch_output = torch_model(*[torch.from_numpy(t).long() for t in inputs])
    torch_loss = l1_lambda * torch.norm(torch_output, 1)
    torch_loss.backward()
    optim.step()

    check_tensors([torch_output.detach().numpy()], outputs, margin=1e-5)

    check_model(torch_model,
                post_proto,
                TORCH_TO_ONNX,
                transform=TRANSPOSE_WEIGHTS)
