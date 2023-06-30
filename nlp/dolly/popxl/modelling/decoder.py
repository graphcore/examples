# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict

import popxl
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.layers import LayerNorm

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer as HFModel

from config import DollyConfig
from .attention import DollySelfAttentionTP
from .feed_forward import DollyFeedForwardTP


class DollyDecoderBlockTP(addons.Module):
    def __init__(self, config: DollyConfig):
        super().__init__()
        self.config = config
        # begins with identical computations: layer norm ln_1
        self.ln_1 = LayerNorm()
        self.ln_2 = LayerNorm()
        # attention is sharded
        # identical computation for bias and skip connection
        self.attention = DollySelfAttentionTP(self.config)
        # begins with identical computations: layer norm ln_2
        # feed forward is sharded
        # identical computation for bias, dropout and skip connection
        self.feed_forward = DollyFeedForwardTP(self.config)

    def build(self, x: popxl.Tensor):
        residual = x
        attn_out = self.attention(self.ln_1(x))

        ff_out = self.feed_forward(self.ln_2(x))
        x = attn_out + ff_out + residual
        return x

    @staticmethod
    def hf_mapping(config: DollyConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.ln_1.weight: to_numpy(hf_model.input_layernorm.weight.data, dtype),
            variables.ln_1.bias: to_numpy(hf_model.input_layernorm.bias.data, dtype),
            variables.ln_2.weight: to_numpy(hf_model.post_attention_layernorm.weight.data, dtype),
            variables.ln_2.bias: to_numpy(hf_model.post_attention_layernorm.bias.data, dtype),
        }
        weights.update(DollySelfAttentionTP.hf_mapping(config, variables.attention, hf_model.attention))
        weights.update(DollyFeedForwardTP.hf_mapping(config, variables.feed_forward, hf_model.mlp))

        return weights


class DollyDecoderTP(addons.Module):
    def __init__(self, config: DollyConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor):

        facts, graph = DollyDecoderBlockTP(self.config).create_graph(x)  # Outline GPT Layer

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x)

        return x
