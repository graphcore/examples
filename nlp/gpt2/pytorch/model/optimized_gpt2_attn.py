# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import poptorch
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import numpy as np


class OptimizedGPT2Attention(GPT2Attention):
    def optimized_attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights *= 1. / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length -
                                    query_length: key_length, :key_length]
            attn_weights -= 1e4 * (1 - causal_mask)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        # attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(
                hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self.optimized_attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class OptimizedGPT2AttentionBuffer(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=False, layer_idx=None)
        self.config = config
        self.register_buffer("past_key",
                             torch.zeros(config.batch, config.n_head, config.seq, int(config.n_embd/config.n_head)))
        self.register_buffer("past_value",
                             torch.zeros(config.batch, config.n_head, config.seq, int(config.n_embd/config.n_head)))

    def optimized_attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights *= 1. / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        else:
            query_length, key_length = query.size(-2), key.size(-2)
            # causal_mask = torch.tril(torch.ones((query_length, key_length)))
            causal_mask = self.bias[:, :, key_length -
                                    query_length: key_length, :key_length]
            attn_weights -= 1e4 * (1 - causal_mask)

        attn_weights = attn_weights.type(value.dtype)
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(
                hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if attention_mask is not None:
            key = torch.cat((key, self.past_key[:, :, :-1, :]), dim=-2)
            value = torch.cat((value, self.past_value[:, :, :-1, :]), dim=-2)
            self.past_key.copy_(key)
            self.past_value.copy_(value)
        else:
            key_ = torch.cat(
                (key, self.past_key[:, :, :-self.config.input_len, :]), dim=-2)
            value_ = torch.cat(
                (value, self.past_value[:, :, :-self.config.input_len, :]), dim=-2)
            self.past_key.copy_(key_)
            self.past_value.copy_(value_)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        attn_output, attn_weights = self.optimized_attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class OptimizedGPT2AttentionCache(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=False, layer_idx=None)
        self.config = config

    def optimized_attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights *= 1. / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        else:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length -
                                    query_length: key_length, :key_length]
            attn_weights -= 1e4 * (1 - causal_mask)

        attn_weights = attn_weights.type(value.dtype)
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # custom op to optimize the layout for attn_weights and value for latency
        attn_weights = attn_weights.type(value.dtype)
        attn_weights_remap = poptorch.custom_op([attn_weights],
                                                "RemapCE",
                                                "ai.graphcore",
                                                1,
                                                example_outputs=[attn_weights],
                                                attributes={"grain_size": 8})
        value_remap = poptorch.custom_op([value],
                                         "RemapCE",
                                         "ai.graphcore",
                                         1,
                                         example_outputs=[value],
                                         attributes={"grain_size": 8})
        attn_weights = attn_weights_remap[0].type(value.dtype)
        value = value_remap[0].type(value.dtype)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(
                hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((key, past_key[:, :, :-1, :]), dim=-2)
            value = torch.cat((value, past_value[:, :, :-1, :]), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self.optimized_attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
