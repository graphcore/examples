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

import torch
import popart
import numpy as np
import pytest

from bert_model import Bert, BertConfig
from tests.torch_bert import BertConfig as TorchBertConfig, BertForNextSentencePrediction

from .full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full nsp graph.
'''
NSP_MAPPING = {
    "bert.pooler.dense.weight": "NSP/PoolW",
    "bert.pooler.dense.bias": "NSP/PoolB",
    "cls.seq_relationship.weight": "NSP/NspW",
    "cls.seq_relationship.bias": "NSP/NspB"
}

NSP_TRANSFORM = {
    "bert.pooler.dense.weight": np.transpose,
    "cls.seq_relationship.weight": np.transpose
}


def test_nsp_fwd(custom_ops):
    #  ------------------- PopART --------------------
    config = BertConfig(task="NSP",
                        vocab_length=9728,
                        num_layers=2,
                        micro_batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        inference=True,
                        no_mask=True,
                        mask_tokens=0,
                        split_qkv=False)
    popart_model = Bert(config)


    #  ------------------- PyTorch -------------------------
    torch_model = BertForNextSentencePrediction(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act=config.activation_type,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        num_labels=2))

    fwd_graph(popart_model,
              torch_model,
              NSP_MAPPING,
              transform=NSP_TRANSFORM)


@pytest.mark.parametrize("opt_type", ["SGD", "LAMB"])
def test_nsp_bwd(custom_ops, opt_type):
    #  ------------------- PopART --------------------
    config = BertConfig(task="NSP",
                        vocab_length=2432,
                        micro_batch_size=1,
                        hidden_size=288,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        no_mask=True,
                        update_embedding_dict=True,
                        split_qkv = (opt_type == "LAMB"))
    popart_model = Bert(config)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForNextSentencePrediction(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act=config.activation_type,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        update_embedding_dict=True,
                        num_labels=2))
    l1_lambda = 0.1

    def popart_loss_fn(outputs):
        loss = popart_model.builder.aiGraphcore.l1loss([outputs[0]], l1_lambda,
                                                       debugContext="l1LossVal",
                                                       reduction=popart.ReductionType.Sum)
        popart_model.builder.virtualGraph(loss, popart_model.nsp_scope.virtualGraph)
        return loss

    def torch_loss_fn(outputs):
        return l1_lambda * torch.norm(outputs[0], 1)

    bwd_graph(popart_model,
              torch_model,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=torch_loss_fn,
              mapping=NSP_MAPPING,
              transform=NSP_TRANSFORM,
              opt_type=opt_type)
