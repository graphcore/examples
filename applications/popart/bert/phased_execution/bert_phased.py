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

import phased_execution.nn as nn
from bert_model import ExecutionMode
from phased_execution.bert_layers import (Attention, FeedForward, MaskLM,
                                          NextSentencePred, SquadProjection)
from phased_execution.bert_layers_serialised import (AttentionSplitHidden,
                                                     BertEmbedding,
                                                     FeedForwardSplitHidden,
                                                     MaskLMSerialised)


class BertEncoder(nn.Block):
    def __init__(self, config, mode, weight_transposed, **kwargs):
        scope_provider = kwargs['scope_provider']
        super(BertEncoder, self).__init__(scope_provider.get_scope('Encoder'), **kwargs)
        self.config = config

        self.embedding = BertEmbedding(
            config.vocab_length,
            config.hidden_size,
            config.sequence_length,
            config.max_positional_length,
            config.embedding_serialization_vocab_steps,
            config.layer_norm_eps,
            not config.no_dropout,
            config.dropout_prob,
            mode,
            detach=not config.update_embedding_dict,
            weight_transposed=weight_transposed,
            custom=True if config.task == 'PRETRAINING' else False,
            **kwargs)

        attention_params = {
            'hidden_size': config.hidden_size,
            'num_heads': config.attention_heads,
            'serialize_matmul': config.split_linear_layers,
            'available_memory_proportion': config.available_memory_proportion[0],
            'epsilon': config.layer_norm_eps,
            'dropout': not config.no_dropout,
            'dropout_prob': config.dropout_prob,
            'attn_dropout': not config.no_attn_dropout,
            'attn_dropout_prob': config.attn_dropout_prob,
            'batch_size': config.batch_size,
            'sequence_length': config.sequence_length,
            'task': config.task,
            'num_mask_tokens': config.mask_tokens,
            'use_default_mem_proportion': config.use_default_available_memory_proportion
        }

        blks = []
        if config.num_attention_splits > 1:
            for i in range(config.num_layers):
                blks.append(
                    (AttentionSplitHidden(
                        f'Layer{i}/Attention',
                        num_splits=config.num_attention_splits,
                        **attention_params,
                        **kwargs),
                     FeedForwardSplitHidden(f'Layer{i}/FF',
                                            num_splits=config.num_ffwd_splits,
                                            input_size=config.hidden_size,
                                            ff_size=config.ff_size,
                                            dropout=not config.no_dropout,
                                            dropout_prob=config.dropout_prob,
                                            epsilon=config.layer_norm_eps,
                                            available_memory_proportion=config.available_memory_proportion[0],
                                            use_default_memory_proportion=config.use_default_available_memory_proportion,
                                            **kwargs)))
        else:
            for i in range(config.num_layers):
                blks.append(
                    (Attention(f'Layer{i}/Attention',
                               input_size=config.hidden_size,
                               split_qkv=config.split_qkv,
                               **attention_params,
                               **kwargs),
                     FeedForward(f'Layer{i}/FF',
                                 config.hidden_size,
                                 config.ff_size,
                                 dropout=not config.no_dropout,
                                 dropout_prob=config.dropout_prob,
                                 epsilon=config.layer_norm_eps,
                                 intermediate_act_func=config.activation_type,
                                 increment_scope=config.split_transformer,
                                 available_memory_proportion=config.available_memory_proportion[0],
                                 use_default_memory_proportion=config.use_default_available_memory_proportion,
                                 **kwargs)))
        self.blks = blks
        self.total_execution_phases = self.total_phases()

    def forward(self, indices, positions, segments, masks=None):
        # Size of act = [batch_size * seq_len, hidden_size]
        act = self.embedding(indices, positions, segments)
        for i, (attention_blk, ffwd_blk) in enumerate(self.blks):
            mask_phase = 0 if self.config.phased_execution_type == 'SINGLE' else i % 2
            mask_reuse_from_layer = 1 if self.config.phased_execution_type == 'SINGLE' else 2
            if (i >= mask_reuse_from_layer):
                if isinstance(attention_blk, Attention):
                    attention_blk.mask = self.blks[mask_phase][0].mask
                else:  # split attention
                    attention_blk.layers[0].mask = self.blks[mask_phase][0].layers[0].mask
            act = attention_blk(act, masks)
            act = ffwd_blk(act)
        return act


class BertModel(nn.Block):
    def __init__(self, config, **kwargs):
        self.config = config
        scope_provider = kwargs['scope_provider']
        super().__init__(scope_provider.get_scope('BertModel'), **kwargs)
        self.weight_transposed = True if config.task == 'PRETRAINING' else False
        self.encoder = BertEncoder(config,
                                   ExecutionMode.PHASED,
                                   weight_transposed=self.weight_transposed,
                                   dtype=config.dtype,
                                   **kwargs)

        if config.task == "PRETRAINING":
            if config.embedding_serialization_vocab_steps > 1:
                projection_weights = []
                for layer in self.encoder.embedding.token_embedding.layers:
                    projection_weights.append(layer.params[0])
                self.mlm = MaskLMSerialised(
                    config.embedding_serialization_vocab_steps,
                    config.vocab_length,
                    config.hidden_size,
                    config.sequence_length,
                    config.batch_size,
                    config.mask_tokens,
                    projection_weights,
                    config.activation_type,
                    dtype=config.dtype,
                    no_cls_layer=config.no_cls_layer,
                    epsilon=config.layer_norm_eps,
                    projection_bias=config.projection_bias,
                    **kwargs)
                self.mlm_scope = self.mlm.concat_scope
            else:
                projection_weights = self.encoder.embedding.token_embedding.params[0]
                self.mlm = MaskLM('MLM',
                                  config.vocab_length,
                                  config.hidden_size,
                                  config.sequence_length,
                                  config.batch_size,
                                  config.mask_tokens,
                                  projection_weights,
                                  config.activation_type,
                                  dtype=config.dtype,
                                  no_cls_layer=config.no_cls_layer,
                                  epsilon=config.layer_norm_eps,
                                  projection_bias=config.projection_bias,
                                  **kwargs)
                self.mlm_scope = self.mlm.scope

        if config.task in ("NSP", "PRETRAINING"):
            self.nsp = NextSentencePred('NSP',
                                        config.batch_size,
                                        config.sequence_length,
                                        config.hidden_size,
                                        config.mask_tokens,
                                        dtype=config.dtype,
                                        **kwargs)
            self.nsp_scope = self.nsp.scope

        elif config.task == "SQUAD":
            self.squad_projection = SquadProjection('Squad',
                                                    config.batch_size,
                                                    config.sequence_length,
                                                    config.hidden_size,
                                                    dtype=config.dtype,
                                                    **kwargs)
            self.squad_scope = self.squad_projection.scope
        self.final_loss_scope = scope_provider.get_scope("FinalLoss", "prev")
        self.total_execution_phases = self.total_phases()

    # TODO: T22392
    def get_model_embeddings(self):
        return [None, None]

    def forward(self, indices, position, segments, masks=None):
        encoded_x = self.encoder(indices, position, segments, masks)

        if self.config.task == "PRETRAINING":
            mlm_logits = self.mlm(encoded_x)
            nsp_logits = self.nsp(encoded_x)
            output = [mlm_logits, nsp_logits]
        elif self.config.task == "NSP":
            output = [self.nsp(encoded_x)]
        elif self.config.task == "SQUAD":
            output = self.squad_projection(encoded_x)

        self.summary()
        return output
