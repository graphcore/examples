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

from bert_model import BertConfig, ExecutionMode, get_model
from tests.torch_bert import BertConfig as TorchBertConfig, BertForNextSentencePrediction

from .full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full nsp graph.
'''
NSP_MAPPING = {}

NSP_MAPPING[ExecutionMode.DEFAULT] = {
    "bert.pooler.dense.weight": "NSP/PoolW",
    "bert.pooler.dense.bias": "NSP/PoolB",
    "cls.seq_relationship.weight": "NSP/NspW",
    "cls.seq_relationship.bias": "NSP/NspB"
}

NSP_MAPPING[ExecutionMode.PHASED] = {
    "bert.pooler.dense.weight": "BertModel/NSP/Pool/Dense/Weight",
    "bert.pooler.dense.bias": "BertModel/NSP/Pool/Dense/Bias",
    "cls.seq_relationship.weight": "BertModel/NSP/Classifier/Dense/Weight",
    "cls.seq_relationship.bias": "BertModel/NSP/Classifier/Dense/Bias"
}

NSP_TRANSFORM = {
    "bert.pooler.dense.weight": np.transpose,
    "cls.seq_relationship.weight": np.transpose
}

test_modes = [ExecutionMode.DEFAULT, pytest.param(ExecutionMode.PHASED, marks=pytest.mark.requires_remote_buffers)]


@pytest.mark.parametrize("mode", test_modes)
def test_nsp_fwd(custom_ops, mode):
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
                        execution_mode=mode,
                        mask_tokens=0,
                        split_qkv=False)
    popart_model = get_model(config, mode)


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
              mode,
              NSP_MAPPING[mode],
              transform=NSP_TRANSFORM)


@pytest.mark.sanity
@pytest.mark.parametrize("mode", test_modes)
@pytest.mark.parametrize("opt_type", ["SGD", "LAMB"])
def test_nsp_bwd(custom_ops, mode, opt_type):
    nsp_bwd(custom_ops, mode, opt_type, 2432, 288)


def nsp_bwd(custom_ops, mode, opt_type, vocab_length=9728, hidden_size=768):
    if mode == ExecutionMode.PHASED:
        # Phased Execution requires atleast two transformer layers to ensure mlm and embedding are in the same virtual graph.
        num_layers = 2
    else:
        num_layers = 1

    #  ------------------- PopART --------------------
    config = BertConfig(task="NSP",
                        vocab_length=vocab_length,
                        num_layers=num_layers,
                        micro_batch_size=1,
                        hidden_size=hidden_size,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        no_mask=True,
                        update_embedding_dict=True,
                        phased_execution_type="single",
                        execution_mode=mode,
                        split_qkv = (opt_type == "LAMB"))
    popart_model = get_model(config, mode)

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
        if mode == ExecutionMode.PHASED:
            with popart_model.scope_provider(popart_model.builder, popart_model.nsp_scope):
                loss = popart_model.builder.aiGraphcore.l1loss([outputs[0]],
                                                               l1_lambda, debugContext="l1LossVal",
                                                               reduction=popart.ReductionType.Sum)
        else:
            loss = popart_model.builder.aiGraphcore.l1loss([outputs[0]], l1_lambda,
                                                           debugContext="l1LossVal",
                                                           reduction=popart.ReductionType.Sum)
            popart_model.builder.virtualGraph(loss, popart_model.nsp_scope.virtualGraph)
        return loss

    def torch_loss_fn(outputs):
        return l1_lambda * torch.norm(outputs[0], 1)

    bwd_graph(popart_model,
              torch_model,
              mode,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=torch_loss_fn,
              mapping=NSP_MAPPING[mode],
              transform=NSP_TRANSFORM,
              opt_type=opt_type)
