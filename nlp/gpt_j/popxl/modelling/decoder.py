# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
from utils.utils import shard
from popxl_addons.layers import Linear, LayerNorm
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)

from transformers.models.gptj.modeling_gptj import GPTJBlock as HFModel
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTJConfigHF

from config import GPTJConfig
from .attention import GPTJSelfAttentionTP
from .feed_forward import GPTJFeedForwardTP


class GPTJDecoderBlockTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config
        # begins with identical computations: layer norm ln_1
        self.ln_1 = LayerNorm()
        # attention is sharded
        # identical computation for bias and skip connection
        self.attention = GPTJSelfAttentionTP(self.config)
        # begins with identical computations: layer norm ln_2
        # feed forward is sharded
        # identical computation for bias, dropout and skip connection
        self.feed_forward = GPTJFeedForwardTP(self.config)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None):
        attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)

        residual = x
        hidden_states = self.ln_1(x)
        attn_out = self.attention(hidden_states, seed=attention_seed)
        ff_out = self.feed_forward(hidden_states, seed=seed)
        x = attn_out + ff_out + residual
        return x

    @staticmethod
    def hf_mapping(config: GPTJConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.ln_1.weight: to_numpy(hf_model.ln_1.weight.data, dtype),
            variables.ln_1.bias: to_numpy(hf_model.ln_1.bias.data, dtype),
        }
        weights.update(GPTJSelfAttentionTP.hf_mapping(config, variables.attention, hf_model.attn))
        weights.update(GPTJFeedForwardTP.hf_mapping(config, variables.feed_forward, hf_model.mlp))

        return weights

    @staticmethod
    def to_hf(config: GPTJConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        attn = GPTJSelfAttentionTP.to_hf(config, variables_data.attention, hf_model.attn)
        mlp = GPTJFeedForwardTP.to_hf(config, variables_data.feed_forward, hf_model.mlp)
        state_dict = {}
        state_dict["ln_1.weight"] = torch.tensor(variables_data.ln_1.weight, dtype=config.torch_dtype)
        state_dict["ln_1.bias"] = torch.tensor(variables_data.ln_1.bias, dtype=config.torch_dtype)
        state_dict.update({"attn." + k: v for k, v in attn.items()})
        state_dict.update({"mlp." + k: v for k, v in mlp.items()})
        return state_dict


class GPTJDecoderTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor):

        facts, graph = GPTJDecoderBlockTP(self.config).create_graph(x)  # Outline GPT Layer

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x)

        return x
