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

from bert_model import BertConfig, ExecutionMode, get_model
from tests.torch_bert import BertConfig as TorchBertConfig, BertForMaskedLM

from tests.utils import requires_remote_buffers

from tests.unit.pytorch.full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full pretraining graph.
'''
ONNX_TORCH_MAPPING = {}
ONNX_TORCH_MAPPING[ExecutionMode.DEFAULT] = {
    "cls.transform.dense.weight": "CLS/LMPredictionW",
    "cls.transform.dense.bias": "CLS/LMPredictionB",
    "cls.transform.LayerNorm.weight": "CLS/Gamma",
    "cls.transform.LayerNorm.bias": "CLS/Beta",
}
ONNX_TORCH_MAPPING[ExecutionMode.PHASED] = {
    "cls.transform.dense.weight": "BertModel/MLM/LMPrediction/Dense/Weight",
    "cls.transform.dense.bias": "BertModel/MLM/LMPrediction/Dense/Bias",
    "cls.transform.LayerNorm.weight": "BertModel/MLM/LMPrediction/Norm/Gamma",
    "cls.transform.LayerNorm.bias": "BertModel/MLM/LMPrediction/Norm/Beta",
}


onnx_torch_tform = {
    "bert.embeddings.word_embeddings.weight": np.transpose,
    "cls.transform.dense.weight": np.transpose,
}


@pytest.mark.parametrize("mode, replication_factor, replicated_weight_sharding",
                         [(ExecutionMode.DEFAULT, 1, False),
                          requires_remote_buffers(ExecutionMode.PHASED, 1, False),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, True),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, False)])
def test_pretraining_fwd(custom_ops, mode, replication_factor, replicated_weight_sharding):
    #  ------------------- PopART --------------------
    config = BertConfig(task="PRETRAINING",
                        vocab_length=9728,
                        num_layers=2,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        activation_type="relu",
                        no_dropout=True,
                        no_attn_dropout=True,
                        no_cls_layer=False,
                        inference=True,
                        no_mask=True,
                        execution_mode=mode,
                        split_qkv=False)

    popart_model = get_model(config, mode)

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

    fwd_graph(popart_model, torch_model, mode, mapping=ONNX_TORCH_MAPPING[mode], transform=onnx_torch_tform,
              replication_factor=replication_factor,
              replicated_weight_sharding=replicated_weight_sharding)


@pytest.mark.parametrize("mode, replication_factor, replicated_weight_sharding, opt_type",
                         [(ExecutionMode.DEFAULT, 1, False, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 1, False, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, True, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, False, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, False, "LAMB"),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, True, "LAMB"),
                          ])
def test_pretraining_bwd(custom_ops, mode, replication_factor, replicated_weight_sharding, opt_type):
    pretraining_bwd(custom_ops, mode, replication_factor, replicated_weight_sharding, opt_type)


@pytest.mark.sanity
@pytest.mark.parametrize("mode, replication_factor, replicated_weight_sharding, opt_type",
                         [(ExecutionMode.DEFAULT, 1, False, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, True, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, True, "LAMB"),
                          ])
def test_pretraining_bwd_sanity(custom_ops, mode, replication_factor, replicated_weight_sharding, opt_type):
    pretraining_bwd(custom_ops, mode, replication_factor, replicated_weight_sharding, opt_type, 2432, 288)


def pretraining_bwd(custom_ops, mode, replication_factor, replicated_weight_sharding, opt_type, vocab_length=9728, hidden_size=768):
    #  ------------------- PopART --------------------
    if mode == ExecutionMode.PHASED:
        # Phased Execution requires atleast two transformer layers to ensure mlm and embedding are in the same virtual graph.
        num_layers = 2
    else:
        num_layers = 1
    config = BertConfig(task="PRETRAINING",
                        vocab_length=vocab_length,
                        num_layers=num_layers,
                        batch_size=1,
                        hidden_size=hidden_size,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        activation_type="relu",
                        no_dropout=True,
                        no_attn_dropout=True,
                        update_embedding_dict=True,
                        no_cls_layer=True,
                        no_mask=True,
                        phased_execution_type="single",
                        execution_mode=mode,
                        split_qkv = (opt_type == "LAMB"))
    popart_model = get_model(config, mode)

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
        if mode == ExecutionMode.PHASED:
            with popart_model.scope_provider(popart_model.builder, popart_model.mlm_scope):
                loss = popart_model.builder.aiGraphcore.l1loss([logits[0]],
                                                               l1_lambda, debugPrefix="l1LossVal",
                                                               reduction=popart.ReductionType.Sum)
        else:
            loss = popart_model.builder.aiGraphcore.l1loss([logits[0]], l1_lambda, debugPrefix="l1LossVal", reduction=popart.ReductionType.Sum)
            popart_model.builder.virtualGraph(loss, popart_model.mlm_scope.virtualGraph)
        return loss

    bwd_graph(popart_model,
              torch_model,
              mode,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=lambda logits: l1_lambda * torch.norm(logits[0], 1),
              mapping={}, transform=onnx_torch_tform,
              replication_factor=replication_factor,
              replicated_weight_sharding=replicated_weight_sharding,
              opt_type=opt_type)
