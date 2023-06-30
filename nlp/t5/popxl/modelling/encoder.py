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
from transformers.models.t5.modeling_t5 import T5Stack as HFStackModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF

from config import T5Config
from .attention import T5SelfAttentionTP
from .feed_forward import T5FeedForwardTP
from .layer_norm import T5LayerNorm


class T5EncoderBlockTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        # begins with identical computations: layer norm ln_1
        self.ln_1 = T5LayerNorm(self.config)
        # attention is sharded
        # identical computation for bias and skip connection
        self.attention = T5SelfAttentionTP(self.config)
        # begins with identical computations: layer norm ln_2
        self.ln_2 = T5LayerNorm(self.config)
        # feed forward is sharded
        # identical computation for bias, dropout and skip connection
        self.feed_forward = T5FeedForwardTP(self.config)

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        rel_pos_weight: Optional[popxl.Tensor] = None,
        seed: Optional[popxl.Tensor] = None,
    ):
        attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)

        hidden_states = self.ln_1(x)
        hidden_states = self.attention(hidden_states, mask, rel_pos_weight, seed=attention_seed)
        x = hidden_states + x

        hidden_states = self.ln_2(x)
        hidden_states = self.feed_forward(hidden_states, seed=seed)
        x = hidden_states + x
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.ln_1.weight: to_numpy(hf_model.layer[0].layer_norm.weight.data, dtype),
            variables.ln_2.weight: to_numpy(hf_model.layer[1].layer_norm.weight.data, dtype),
        }
        weights.update(T5SelfAttentionTP.hf_mapping(config, variables.attention, hf_model.layer[0].SelfAttention))
        weights.update(T5FeedForwardTP.hf_mapping(config, variables.feed_forward, hf_model.layer[1].DenseReluDense))
        return weights

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        attn = T5SelfAttentionTP.to_hf(config, variables_data.attention, hf_model.layer[0])
        mlp = T5FeedForwardTP.to_hf(config, variables_data.feed_forward, hf_model.layer[1])
        state_dict = {}
        state_dict["layer.0.layer_norm.weight"] = torch.tensor(variables_data.ln_1.weight, dtype=config.torch_dtype)
        state_dict.update({"layer.0.SelfAttention." + k: v for k, v in attn.items()})
        state_dict["layer.1.layer_norm.weight"] = torch.tensor(variables_data.ln_2.weight, dtype=config.torch_dtype)
        state_dict.update({"layer.1.DenseReluDense." + k: v for k, v in mlp.items()})
        return state_dict


class T5EncoderHead(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        # layer norm at the end of the encoder stack
        self.ln_f = T5LayerNorm(self.config)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        x = self.ln_f(x)
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFStackModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.ln_f.weight: to_numpy(hf_model.final_layer_norm.weight.data, dtype),
        }
        return weights

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFStackModel) -> Dict[str, torch.Tensor]:
        state_dict = {}
        ln = T5LayerNorm.to_hf(config, variables_data.ln_f, hf_model.final_layer_norm)
        state_dict.update({"encoder.final_layer_norm." + k: v for k, v in ln.items()})
        return state_dict


class T5EncoderTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor, mask: popxl.Tensor):
        # First encoder layer
        facts, graph = T5EncoderBlockTP(self.config).create_graph(x.spec, mask.spec)
        first_enc_vars = self.add_variable_inputs(0, facts)
        (x,) = graph.bind(first_enc_vars).call(x, mask)

        # Following encoder layers
        rel_pos_weight = first_enc_vars.attention.heads.rel_pos_embedding.weight
        facts, graph = T5EncoderBlockTP(self.config).create_graph(x.spec, mask.spec, rel_pos_weight.spec)

        for i in range(self.config.model.layers - 1):
            args_nt = self.add_variable_inputs(i + 1, facts)
            (x,) = graph.bind(args_nt).call(x, mask, rel_pos_weight)

        return x
