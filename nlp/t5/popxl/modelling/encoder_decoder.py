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
from .attention import T5SelfAttentionTP, T5CrossAttentionTP
from .feed_forward import T5FeedForwardTP
from .layer_norm import T5LayerNorm


class T5BlockTP(addons.Module):
    """A unified block, that acts as an encoder block or
    decoder block, depending on the inputs given."""

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
        # attention is sharded
        # identical computation for bias and skip connection
        self.cross_attention = T5CrossAttentionTP(self.config)
        # begins with identical computations: layer norm ln_3
        self.ln_3 = T5LayerNorm(self.config)
        # feed forward is sharded
        # identical computation for bias, dropout and skip connection
        self.feed_forward = T5FeedForwardTP(self.config)
        self.scale_ff = config.model.scale_ff

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        enc_output: popxl.Tensor,
        enc_mask: popxl.Tensor,
        cross_attention_scale: popxl.Tensor,
        rel_pos_weight: Optional[popxl.Tensor] = None,
        seed: Optional[popxl.Tensor] = None,
    ):
        attention_seed = None
        cross_attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)
            seed, cross_attention_seed = ops.split_random_seed(seed)

        hidden_states = self.ln_1(x)
        hidden_states = self.attention(
            hidden_states, mask, 1 - cross_attention_scale, rel_pos_weight, seed=attention_seed
        )
        if x.dtype == popxl.float32 and self.scale_ff > 1:
            x = ops.cast(x, popxl.float16)
        x = hidden_states + x

        hidden_states = self.ln_2(x)
        if enc_output.dtype == popxl.float32 and self.scale_ff > 1:
            enc_output = ops.cast(enc_output, popxl.float16)
        hidden_states = self.cross_attention(hidden_states, enc_mask, enc_output, seed=cross_attention_seed)
        # The encoder will mask out the cross-attention part
        hidden_states = cross_attention_scale * hidden_states
        x = hidden_states + x

        hidden_states = self.ln_3(x)
        hidden_states = self.feed_forward(hidden_states, seed=seed)
        if self.scale_ff > 1:
            x = ops.cast(x, popxl.float32)
            # Undo the scale down done in the feed forward block, but now safely in fp32
            hidden_states = ops.cast(hidden_states, popxl.float32) * self.scale_ff
        x = hidden_states + x
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        # Function that takes a tensor and return a numpy array of zeros of the appropriate shape
        def zero_arr(var: popxl.Tensor, n_shards: int, np_dtype: np.dtype) -> np.ndarray:
            shape = var.shape
            if n_shards > 1:
                shape = (n_shards,) + shape
            return np.zeros(shape, np_dtype)

        dtype = config.model.dtype
        if hf_model.is_decoder:
            attn_hf, crossattn_hf, mlp_hf = hf_model.layer
        else:
            attn_hf, mlp_hf = hf_model.layer
            crossattn_hf = None
        weights = {}
        weights[variables.ln_1.weight] = to_numpy(hf_model.layer[0].layer_norm.weight.data, dtype)
        weights.update(T5SelfAttentionTP.hf_mapping(config, variables.attention, attn_hf.SelfAttention))
        idx = 1
        if crossattn_hf is None:
            np_dtype = dtype.as_numpy()
            # For the encoder, we fill the weights with zeros
            crossattn_vars = [
                variables.ln_2.weight,
                variables.cross_attention.heads.q.weight,
                variables.cross_attention.heads.kv.weight,
                variables.cross_attention.output.weight,
            ]
            for i, var in enumerate(crossattn_vars):
                n_shards = 1 if i == 0 else config.execution.tensor_parallel
                weights[var] = zero_arr(var, n_shards, np_dtype)
        else:
            weights[variables.ln_2.weight] = to_numpy(hf_model.layer[1].layer_norm.weight.data, dtype)
            weights.update(
                T5CrossAttentionTP.hf_mapping(config, variables.cross_attention, crossattn_hf.EncDecAttention)
            )
            idx += 1
        weights[variables.ln_3.weight] = to_numpy(hf_model.layer[idx].layer_norm.weight.data, dtype)
        weights.update(T5FeedForwardTP.hf_mapping(config, variables.feed_forward, mlp_hf.DenseReluDense))
        return weights

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        if hf_model.is_decoder:
            attn_hf, crossattn_hf, mlp_hf = hf_model.layer
        else:
            attn_hf, mlp_hf = hf_model.layer
            crossattn_hf = None
        state_dict = {}
        state_dict["layer.0.layer_norm.weight"] = torch.tensor(variables_data.ln_1.weight, dtype=config.torch_dtype)
        attn = T5SelfAttentionTP.to_hf(config, variables_data.attention, attn_hf.SelfAttention)
        state_dict.update({"layer.0.SelfAttention." + k: v for k, v in attn.items()})
        idx = 1
        if crossattn_hf is not None:
            state_dict["layer.1.layer_norm.weight"] = torch.tensor(variables_data.ln_2.weight, dtype=config.torch_dtype)
            crossattn = T5CrossAttentionTP.to_hf(config, variables_data.cross_attention, crossattn_hf.EncDecAttention)
            state_dict.update({"layer.1.EncDecAttention." + k: v for k, v in crossattn.items()})
            idx += 1
        state_dict[f"layer.{idx}.layer_norm.weight"] = torch.tensor(
            variables_data.ln_3.weight, dtype=config.torch_dtype
        )
        mlp = T5FeedForwardTP.to_hf(config, variables_data.feed_forward, mlp_hf.DenseReluDense)
        state_dict.update({f"layer.{idx}.DenseReluDense." + k: v for k, v in mlp.items()})
        return state_dict


class T5EncoderDecoderTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        enc_output: popxl.Tensor,
        enc_mask: popxl.Tensor,
        cross_attention_scale: popxl.Tensor,
        rel_pos_weight: popxl.Tensor,
    ):
        facts, graph = T5BlockTP(self.config).create_graph(
            x.spec,
            mask.spec,
            enc_output.spec,
            enc_mask.spec,
            cross_attention_scale.spec,
            rel_pos_weight.spec,
        )

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x, mask, enc_output, enc_mask, cross_attention_scale, rel_pos_weight)
        return x


class T5EncoderHead(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        # layer norm at the end of the encoder stack
        self.ln_f = T5LayerNorm(self.config)
        self.should_upcast = config.model.scale_ff > 1

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        x = self.ln_f(x)
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)

        if self.should_upcast:
            x = ops.cast(x, popxl.float32)
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
