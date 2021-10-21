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
from tests.torch_bert import BertConfig as TorchBertConfig, BertForQuestionAnswering

from tests.utils import sanity

from .full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full squad graph.
'''

ONNX_TORCH_MAPPING = {
    "qa_outputs.weight": "Squad/SquadW",
    "qa_outputs.bias": "Squad/SquadB"
}


def test_squad_fwd(custom_ops):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        encoder_start_ipu=1,
                        vocab_length=1024,
                        micro_batch_size=1,
                        hidden_size=64,
                        attention_heads=2,
                        sequence_length=20,
                        max_positional_length=20,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        inference=True,
                        no_mask=True,
                        split_qkv=False,
                        squad_single_output=False)

    popart_model = Bert(config)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForQuestionAnswering(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=2,
                        num_labels=2))

    fwd_graph(popart_model,
              torch_model,
              mapping=ONNX_TORCH_MAPPING,
              transform={
                  "qa_outputs.weight": np.transpose
              })


@pytest.mark.parametrize("replication_factor, replicated_tensor_sharding, opt_type",
                         [sanity(2, True, "SGD"),
                          sanity(2, False, "LAMB"),
                          (1, False, "SGD"),
                          (2, True, "SGD"),
                          (2, False, "LAMB"),
                          (2, True, "LAMB")])
def test_squad_bwd(custom_ops, replication_factor, replicated_tensor_sharding, opt_type):
    #  ------------------- PopART --------------------
    config = BertConfig(task="SQUAD",
                        num_layers=2,
                        encoder_start_ipu=1,
                        vocab_length=1024,
                        micro_batch_size=1,
                        hidden_size=64,
                        attention_heads=2,
                        sequence_length=20,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        no_attn_dropout=True,
                        update_embedding_dict=True,
                        no_mask=True,
                        split_qkv=(opt_type == "LAMB"))
    popart_model = Bert(config)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForQuestionAnswering(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=2,
                        update_embedding_dict=True,
                        num_labels=2))

    l1_lambda = 0.1

    def popart_loss_fn(outputs):
        losses = [
            popart_model.builder.aiGraphcore.l1loss(
                [outputs[0]], l1_lambda, debugContext="startsLossVal", reduction=popart.ReductionType.Sum),
            popart_model.builder.aiGraphcore.l1loss(
                [outputs[1]], l1_lambda, debugContext="endsLossVal", reduction=popart.ReductionType.Sum),
        ]
        for loss in losses:
            popart_model.builder.virtualGraph(loss, popart_model.squad_scope.virtualGraph)

        final_loss = popart_model.builder.aiOnnx.sum(losses, debugContext="finalLoss")
        popart_model.builder.virtualGraph(final_loss, popart_model.squad_scope.virtualGraph)
        return final_loss

    def torch_loss_fn(outputs):
        torch_losses = [l1_lambda * torch.norm(output, 1)
                        for output in outputs]
        return torch.add(*torch_losses)

    bwd_graph(popart_model,
              torch_model,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=torch_loss_fn,
              mapping=ONNX_TORCH_MAPPING,
              transform={
                  "qa_outputs.weight": np.transpose
              },
              replication_factor=replication_factor,
              replicated_tensor_sharding=replicated_tensor_sharding,
              opt_type=opt_type)
