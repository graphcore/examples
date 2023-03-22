# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict

import popxl
from popxl import ops
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict

from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFModel

from config import GPTConfig
from .attention import GPTSelfAttentionTP, GPTSelfAttentionTP2D
from .feed_forward import GPTFeedForwardTP, GPTFeedForwardTP2D


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

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)
        x = self.attention(x, seed=attention_seed)
        x = self.feed_forward(x, seed=seed)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        weights = GPTSelfAttentionTP.hf_mapping(config, variables.attention, hf_model)
        weights.update(GPTFeedForwardTP.hf_mapping(config, variables.feed_forward, hf_model))
        return weights


class GPTDecoderBlockTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attention = GPTSelfAttentionTP2D(self.config)
        self.feed_forward = GPTFeedForwardTP2D(self.config)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)
        x = self.attention(x, seed=attention_seed)
        x = self.feed_forward(x, seed=seed)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        weights = GPTSelfAttentionTP2D.hf_mapping(config, variables.attention, hf_model)
        weights.update(GPTFeedForwardTP2D.hf_mapping(config, variables.feed_forward, hf_model))
        return weights


class GPTDecoderTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor) -> popxl.Tensor:

        facts, graph = GPTDecoderBlockTP(self.config).create_graph(x)  # Outline GPT Layer

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x)

        return x


class GPTDecoderTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor) -> popxl.Tensor:

        facts, graph = GPTDecoderBlockTP2D(self.config).create_graph(x)  # Outline GPT Layer

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x)

        return x
