# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
from typing import Dict

import popxl
from popxl import ops
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.named_tensors import NamedTensorData
from popxl_addons.array_munging import shard
from popxl_addons.layers import Linear, LayerNorm
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock as HFModel
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTJConfigHF

from configs import GPTJConfig
from .attention import GPTJSelfAttentionTP
from .feed_forward import GPTJFeedForwardTP
from modelling.adapters_TP import AdapterTP


class GPTJDecoderBlockTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config
        # begins with identical computations: layer norm ln_1
        self.ln_1 = LayerNorm()
        # attention is sharded
        # identical computation for bias and skip connection
        self.attention = GPTJSelfAttentionTP(self.config)

        if self.config.ff_adapter.mode:
            a_cfg = self.config.ff_adapter
            self.ff_adapter = AdapterTP(
                config=self.config,
                dim=self.config.hidden_size,
                downsample_factor=a_cfg.downsample_factor,
                add_layernorm=a_cfg.layer_norm,
            )

        # begins with identical computations: layer norm ln_2
        # feed forward is sharded
        # identical computation for bias and skip connection
        self.feed_forward = GPTJFeedForwardTP(self.config)

    def build(self, x: popxl.Tensor):
        residual = x
        hidden_states = self.ln_1(x)

        attn_out = self.attention(hidden_states)

        ff_out = self.feed_forward(hidden_states)

        if self.config.ff_adapter.mode == "normal":
            ff_out = self.ff_adapter(ff_out)
        x = attn_out + ff_out + residual
        return x

    @staticmethod
    def finetuneanon_mapping(
        config: GPTJConfig, variables: NamedTensors, hf_model: HFModel, from_magma: bool = True
    ) -> Dict[popxl.Tensor, np.ndarray]:
        # from magma necessary because magma wraps and extends the layers to account for adapters
        dtype = config.dtype
        weights = {
            variables.ln_1.weight: to_numpy(hf_model.ln_1.weight.data, dtype),
            variables.ln_1.bias: to_numpy(hf_model.ln_1.bias.data, dtype),
        }
        weights.update(GPTJSelfAttentionTP.finetuneanon_mapping(config, variables.attention, hf_model.attn))

        if from_magma:
            weights.update(GPTJFeedForwardTP.finetuneanon_mapping(config, variables.feed_forward, hf_model.mlp[0]))

            if config.ff_adapter.mode is not None:
                weights.update(AdapterTP.magma_mapping(config, hf_model.mlp[1], variables.ff_adapter))
        else:
            assert config.ff_adapter.mode is None
            weights.update(GPTJFeedForwardTP.finetuneanon_mapping(config, variables.feed_forward, hf_model.mlp))

        return weights


class GPTJDecoderTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor):

        facts, graph = GPTJDecoderBlockTP(self.config).create_graph(x)  # Outline GPT Layer

        for i in range(self.config.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x)

        return x
