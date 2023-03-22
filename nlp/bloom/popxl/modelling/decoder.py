# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Dict

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict
from transformers.models.bloom.modeling_bloom import BloomBlock as HFModel

import popxl
from config import BloomConfig

from .attention import BloomSelfAttentionTP2D
from .feed_forward import BloomFeedForwardTP2D


class BloomDecoderBlockTP2D(addons.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.config = config
        self.attention = BloomSelfAttentionTP2D(self.config)
        self.feed_forward = BloomFeedForwardTP2D(self.config)

    def build(self, x: popxl.Tensor):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

    @staticmethod
    def hf_mapping(config: BloomConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        weights = BloomSelfAttentionTP2D.hf_mapping(config, variables.attention, hf_model)
        weights.update(BloomFeedForwardTP2D.hf_mapping(config, variables.feed_forward, hf_model))
        return weights


class BloomDecoderTP2D(addons.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.config = config

    def build(self, x: popxl.Tensor):
        facts, graph = BloomDecoderBlockTP2D(self.config).create_graph(x)  # Outline Bloom Layer

        for i in range(self.config.model.layers):
            args_nt = self.add_variable_inputs(i, facts)
            (x,) = graph.bind(args_nt).call(x)

        return x
