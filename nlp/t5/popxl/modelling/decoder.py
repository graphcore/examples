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

from transformers.models.t5.modeling_t5 import T5Block as HFModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF

from config import T5Config
from .attention import T5SelfAttentionTP, T5CrossAttentionTP
from .feed_forward import T5FeedForwardTP
from .layer_norm import T5LayerNorm


class T5DecoderBlockTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        # begins with identical computations: layer norm ln_1
        self.ln_1 = T5LayerNorm(self.config)
        # attention is sharded
        # identical computation for bias and skip connection
        self.attention = T5SelfAttentionTP(self.config, is_decoder=True)
        # begins with identical computations: layer norm ln_2
        self.ln_2 = T5LayerNorm(self.config)
        # attention is sharded
        # identical computation for bias and skip connection
        self.cross_attention = T5CrossAttentionTP(self.config)
        # begins with identical computations: layer norm ln_3
        self.ln_3 = T5LayerNorm(self.config)
        # feed forward is sharded
        # identical computation for bias, dropout and skip connection
        self.feed_forward = T5FeedForwardTP(self.config)

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        enc_output: popxl.Tensor,
        enc_mask: popxl.Tensor,
        rel_pos_weight: Optional[popxl.Tensor] = None,
        seed: Optional[popxl.Tensor] = None,
    ):
        attention_seed = None
        cross_attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)
            seed, cross_attention_seed = ops.split_random_seed(seed)

        hidden_states = self.ln_1(x)
        hidden_states = self.attention(hidden_states, mask, rel_pos_weight, seed=attention_seed)
        x = hidden_states + x

        hidden_states = self.ln_2(x)
        hidden_states = self.cross_attention(hidden_states, enc_mask, enc_output, seed=cross_attention_seed)
        x = hidden_states + x

        hidden_states = self.ln_3(x)
        hidden_states = self.feed_forward(hidden_states, seed=seed)
        x = hidden_states + x
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.ln_1.weight: to_numpy(hf_model.layer[0].layer_norm.weight.data, dtype),
            variables.ln_2.weight: to_numpy(hf_model.layer[1].layer_norm.weight.data, dtype),
            variables.ln_3.weight: to_numpy(hf_model.layer[2].layer_norm.weight.data, dtype),
        }
        weights.update(T5SelfAttentionTP.hf_mapping(config, variables.attention, hf_model.layer[0].SelfAttention))
        weights.update(
            T5CrossAttentionTP.hf_mapping(config, variables.cross_attention, hf_model.layer[1].EncDecAttention)
        )
        weights.update(T5FeedForwardTP.hf_mapping(config, variables.feed_forward, hf_model.layer[2].DenseReluDense))
        return weights

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        attn = T5SelfAttentionTP.to_hf(config, variables_data.attention, hf_model.layer[0])
        crossattn = T5CrossAttentionTP.to_hf(config, variables_data.cross_attention, hf_model.layer[1])
        mlp = T5FeedForwardTP.to_hf(config, variables_data.feed_forward, hf_model.layer[2])
        state_dict = {}
        state_dict["layer.0.layer_norm.weight"] = torch.tensor(variables_data.ln_1.weight, dtype=config.torch_dtype)
        state_dict.update({"layer.0.SelfAttention." + k: v for k, v in attn.items()})
        state_dict["layer.1.layer_norm.weight"] = torch.tensor(variables_data.ln_2.weight, dtype=config.torch_dtype)
        state_dict.update({"layer.1.EncDecAttention." + k: v for k, v in crossattn.items()})
        state_dict["layer.2.layer_norm.weight"] = torch.tensor(variables_data.ln_3.weight, dtype=config.torch_dtype)
        state_dict.update({"layer.2.DenseReluDense." + k: v for k, v in mlp.items()})
        return state_dict


class T5DecoderTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        enc_output: popxl.Tensor,
        enc_mask: popxl.Tensor,
    ):
        # First decoder layer
        facts, graph = T5DecoderBlockTP(self.config).create_graph(x.spec, mask.spec, enc_output.spec, enc_mask.spec)
        first_dec_vars = self.add_variable_inputs(0, facts)
        (x,) = graph.bind(first_dec_vars).call(x, mask, enc_output, enc_mask)

        # Following decoder layers
        rel_pos_weight = first_dec_vars.attention.heads.rel_pos_embedding.weight
        facts, graph = T5DecoderBlockTP(self.config).create_graph(
            x.spec, mask.spec, enc_output.spec, enc_mask.spec, rel_pos_weight.spec
        )

        for i in range(self.config.model.layers - 1):
            args_nt = self.add_variable_inputs(i + 1, facts)
            (x,) = graph.bind(args_nt).call(x, mask, enc_output, enc_mask, rel_pos_weight)

        return x
