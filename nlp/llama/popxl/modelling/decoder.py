# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict, Optional

import popxl
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors

from transformers.models.llama.modeling_llama import LlamaDecoderLayer as HFModel

from config import LlamaConfig
from .attention import LlamaSelfAttentionTP
from .feed_forward import LlamaFeedForwardTP
from .rms_norm import LlamaRMSNorm


class LlamaDecoderBlockTP(addons.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        # begins with identical computations: layer norm ln_1
        self.ln_1 = LlamaRMSNorm(self.config)
        # attention is sharded
        # identical computation for bias and skip connection
        self.attention = LlamaSelfAttentionTP(self.config)
        # begins with identical computations: layer norm ln_2
        self.ln_2 = LlamaRMSNorm(self.config)
        # feed forward is sharded
        # identical computation for bias, dropout and skip connection
        self.feed_forward = LlamaFeedForwardTP(self.config)

    def build(
        self,
        x: popxl.Tensor,
        last_token_indices: Optional[popxl.Tensor] = None,
        past_k: Optional[popxl.Tensor] = None,
        past_v: Optional[popxl.Tensor] = None,
    ):

        initial_residual = x
        ax = self.ln_1(x)

        attn_o = self.attention(ax, last_token_indices, past_k, past_v)

        ax = initial_residual + attn_o[0]

        post_attn_residual = ax
        fx = self.ln_2(ax)
        fx = self.feed_forward(fx)

        hs = post_attn_residual + fx

        outputs = (hs,)
        if self.config.execution.use_cache:
            outputs = (
                hs,
                attn_o[1],
                attn_o[2],
            )

        return outputs

    @staticmethod
    def hf_mapping(config: LlamaConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        weights = {
            variables.ln_1.weight: to_numpy(hf_model.input_layernorm.weight.data, dtype),
            variables.ln_2.weight: to_numpy(hf_model.post_attention_layernorm.weight.data, dtype),
        }

        weights.update(LlamaSelfAttentionTP.hf_mapping(config, variables.attention, hf_model.self_attn))
        weights.update(LlamaFeedForwardTP.hf_mapping(config, variables.feed_forward, hf_model.mlp))

        return weights


class LlamaDecoderTP(addons.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor):

        facts, graph = LlamaDecoderBlockTP(self.config).create_graph(x)  # Outline GPT Layer

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x)

        return x
