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
import pytest
import numpy as np

from bert_model import Bert, BertConfig
from tests.torch_bert import BertConfig as TorchBertConfig, BertForMaskedLM
from tests.unit.pytorch.full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full pretraining graph.
'''
ONNX_TORCH_MAPPING = {
    "cls.transform.dense.weight": "CLS/LMPredictionW",
    "cls.transform.dense.bias": "CLS/LMPredictionB",
    "cls.transform.LayerNorm.weight": "CLS/Gamma",
    "cls.transform.LayerNorm.bias": "CLS/Beta",
}


onnx_torch_tform = {
    "bert.embeddings.word_embeddings.weight": np.transpose,
    "cls.transform.dense.weight": np.transpose,
}


def test_pretraining_fwd(custom_ops):
    #  ------------------- PopART --------------------
    config = BertConfig(task="PRETRAINING",
                        encoder_start_ipu=1,
                        vocab_length=1024,
                        micro_batch_size=1,
                        hidden_size=64,
                        attention_heads=2,
                        sequence_length=20,
                        max_positional_length=20,
                        mask_tokens=2,
                        popart_dtype="FLOAT",
                        activation_type="relu",
                        no_dropout=True,
                        no_attn_dropout=True,
                        no_cls_layer=False,
                        inference=True,
                        no_mask=True,
                        split_qkv=False)

    popart_model = Bert(config)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForMaskedLM(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        no_cls_layer=config.no_cls_layer))

    fwd_graph(popart_model, torch_model, mapping=ONNX_TORCH_MAPPING, transform=onnx_torch_tform)


@pytest.mark.parametrize("opt_type", ["SGD", "LAMB"])
def test_pretraining_bwd(custom_ops, opt_type):
    #  ------------------- PopART --------------------
    config = BertConfig(task="PRETRAINING",
                        encoder_start_ipu=1,
                        vocab_length=1024,
                        micro_batch_size=1,
                        hidden_size=64,
                        attention_heads=2,
                        sequence_length=20,
                        max_positional_length=20,
                        mask_tokens=2,
                        popart_dtype="FLOAT",
                        activation_type="relu",
                        no_dropout=True,
                        no_attn_dropout=True,
                        update_embedding_dict=True,
                        no_cls_layer=True,
                        no_mask=True,
                        split_qkv = (opt_type == "LAMB"))
    popart_model = Bert(config)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForMaskedLM(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        update_embedding_dict=True,
                        mask_tokens=config.mask_tokens))

    l1_lambda = 0.1

    def popart_loss_fn(logits):
        loss = popart_model.builder.aiGraphcore.l1loss([logits[0]], l1_lambda, debugContext="l1LossVal", reduction=popart.ReductionType.Sum)
        popart_model.builder.virtualGraph(loss, popart_model.mlm_scope.virtualGraph)
        return loss

    bwd_graph(popart_model,
              torch_model,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=lambda logits: l1_lambda * torch.norm(logits[0], 1),
              mapping={},
              transform=onnx_torch_tform,
              opt_type=opt_type)
