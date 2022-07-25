# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
import math
from functools import partial

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from utils.utils import shard
from popxl_addons.layers import Linear, LayerNorm
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFModel

from config import GPTConfig
from .attention import GPTSelfAttention, GPTSelfAttentionTP
from .feed_forward import GPTFeedForward, GPTFeedForwardTP


class GPTDecoderBlock(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attention = GPTSelfAttention(self.config)
        self.feed_forward = GPTFeedForward(self.config)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None):
        attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)
        # original HF
        # attn_out = self.attention(self.ln_1(x), seed=attention_seed)
        # x = x + attn_out
        # ff_out = self.feed_forward(self.ln_2(x), seed=seed)
        # x = x + ff_out
        x = self.attention(x, seed=attention_seed)  # includes layer norm ln_1 and skip connection
        # includes layern norm ln_2 and skip connection
        x = self.feed_forward(x, seed=seed)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        weights = GPTSelfAttention.hf_mapping(config, variables.attention, hf_model)
        weights.update(GPTFeedForward.hf_mapping(config, variables.feed_forward, hf_model))

        return weights


class GPTDecoderBlockTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # begins with identical computations: layer norm ln_1
        # attention is sharded
        # identical computation for bias and skip connection
        self.attention = GPTSelfAttentionTP(self.config)
        # begins with identical computations: layer norm ln_2
        # feed forward is sharded
        # identical computation for bias, dropout and skip connection
        self.feed_forward = GPTFeedForwardTP(self.config)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None):
        attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)
        x = self.attention(x, seed=attention_seed)
        x = self.feed_forward(x, seed=seed)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        weights = GPTSelfAttentionTP.hf_mapping(config, variables.attention, hf_model)
        weights.update(GPTFeedForwardTP.hf_mapping(config, variables.feed_forward, hf_model))

        return weights


class GPTDecoder(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor):
        facts, graph = GPTDecoderBlock(self.config).create_graph(
            x)  # Outline GPT Layer
        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            x, = graph.bind(args_nt).call(x)
        return x


class GPTDecoderTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor):

        facts, graph = GPTDecoderBlockTP(self.config).create_graph(x)  # Outline GPT Layer

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            x, = graph.bind(args_nt).call(x)

        return x
