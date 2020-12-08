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
from tests.torch_bert import BertConfig as TorchBertConfig, BertForQuestionAnswering

from tests.utils import requires_remote_buffers

from .full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full squad graph.
'''

ONNX_TORCH_MAPPING = {}
ONNX_TORCH_MAPPING[ExecutionMode.DEFAULT] = {
    "qa_outputs.weight": "Squad/SquadW",
    "qa_outputs.bias": "Squad/SquadB"
}
ONNX_TORCH_MAPPING[ExecutionMode.PHASED] = {
    "qa_outputs.weight": "BertModel/Squad/Dense/Weight",
    "qa_outputs.bias": "BertModel/Squad/Dense/Bias"
}


@pytest.mark.parametrize("mode, replication_factor, replicated_weight_sharding",
                         [(ExecutionMode.DEFAULT, 1, False),
                          requires_remote_buffers(ExecutionMode.PHASED, 1, False),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, True),
                          requires_remote_buffers(ExecutionMode.PHASED, 4, False)])
def test_squad_fwd(mode, replication_factor, replicated_weight_sharding):
    split_qkv = False
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        num_layers=2,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        inference=True,
                        no_mask=True,
                        execution_mode=mode,
                        split_qkv=split_qkv,
                        squad_single_output=False)

    popart_model = get_model(config, mode)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForQuestionAnswering(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        num_labels=2))

    fwd_graph(popart_model,
              torch_model,
              mode,
              mapping=ONNX_TORCH_MAPPING[mode],
              transform={
                  "qa_outputs.weight": np.transpose
              },
              replication_factor=replication_factor,
              replicated_weight_sharding=replicated_weight_sharding)


@pytest.mark.parametrize("mode, replication_factor, replicated_weight_sharding, opt_type",
                         [(ExecutionMode.DEFAULT, 1, False, "SGD"),
                          (ExecutionMode.DEFAULT, 2, True, "SGD"),
                          (ExecutionMode.DEFAULT, 2, False, "LAMB"),
                          (ExecutionMode.DEFAULT, 2, True, "LAMB"),
                          requires_remote_buffers(ExecutionMode.PHASED, 1, False, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, False, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, False, "LAMB"),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, True, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 1, False, "LAMB")
                          ])
def test_squad_bwd(mode, replication_factor, replicated_weight_sharding, opt_type):
    squad_bwd(mode, replication_factor, replicated_weight_sharding, opt_type)


@pytest.mark.sanity
@pytest.mark.parametrize("mode, replication_factor, replicated_weight_sharding, opt_type",
                         [(ExecutionMode.DEFAULT, 2, True, "SGD"),
                          (ExecutionMode.DEFAULT, 2, False, "LAMB"),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, False, "SGD"),
                          requires_remote_buffers(ExecutionMode.PHASED, 2, True, "LAMB")
                          ])
def test_squad_bwd_sanity(mode, replication_factor, replicated_weight_sharding, opt_type):
    squad_bwd(mode, replication_factor, replicated_weight_sharding, opt_type, 2432, 288)


def squad_bwd(mode, replication_factor, replicated_weight_sharding, opt_type, vocab_length=9728, hidden_size=768):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        vocab_length=vocab_length,
                        num_layers=1,
                        batch_size=1,
                        hidden_size=hidden_size,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        update_embedding_dict=True,
                        no_mask=True,
                        execution_mode=mode,
                        split_qkv=(opt_type == "LAMB"))
    popart_model = get_model(config, mode)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForQuestionAnswering(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        update_embedding_dict=True,
                        num_labels=2))

    l1_lambda = 0.1

    def popart_loss_fn(outputs):
        if mode == ExecutionMode.PHASED:
            with popart_model.scope_provider(popart_model.builder, popart_model.squad_scope):
                losses = [
                    popart_model.builder.aiGraphcore.l1loss(
                        [outputs[0]], l1_lambda, debugPrefix="startsLossVal", reduction=popart.ReductionType.Sum),
                    popart_model.builder.aiGraphcore.l1loss(
                        [outputs[1]], l1_lambda, debugPrefix="endsLossVal", reduction=popart.ReductionType.Sum),
                ]
                final_loss = popart_model.builder.aiOnnx.sum(losses, debugPrefix="finalLoss")

        else:
            losses = [
                popart_model.builder.aiGraphcore.l1loss(
                    [outputs[0]], l1_lambda, debugPrefix="startsLossVal", reduction=popart.ReductionType.Sum),
                popart_model.builder.aiGraphcore.l1loss(
                    [outputs[1]], l1_lambda, debugPrefix="endsLossVal", reduction=popart.ReductionType.Sum),
            ]
            for loss in losses:
                popart_model.builder.virtualGraph(loss, popart_model.squad_scope.virtualGraph)

            final_loss = popart_model.builder.aiOnnx.sum(losses, debugPrefix="finalLoss")
            popart_model.builder.virtualGraph(final_loss, popart_model.squad_scope.virtualGraph)
        return final_loss

    def torch_loss_fn(outputs):
        torch_losses = [l1_lambda * torch.norm(output, 1)
                        for output in outputs]
        return torch.add(*torch_losses)

    bwd_graph(popart_model,
              torch_model,
              mode,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=torch_loss_fn,
              mapping=ONNX_TORCH_MAPPING[mode],
              transform={
                  "qa_outputs.weight": np.transpose
              },
              replication_factor=replication_factor,
              replicated_weight_sharding=replicated_weight_sharding,
              opt_type=opt_type)
