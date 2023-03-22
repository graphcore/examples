# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Dict

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import shard
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.utils import WeightsDict
from transformers.models.bloom.modeling_bloom import BloomModel as HFModel

import popxl
from config import BloomConfig
from modelling.decoder import BloomDecoderBlockTP2D, BloomDecoderTP2D
from popxl.utils import to_numpy
from utils.utils import tp2d_replica_groups

from .decoder import BloomDecoderTP2D
from .embedding import BloomEmbeddingTP2D


class BloomModelTP2D(addons.Module):
    def __init__(self, config: BloomConfig, include_layer_norm=True):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        # Only embeddings for words, positions are handled using attention bias
        self.embedding = BloomEmbeddingTP2D(self.config)
        self.decoder = BloomDecoderTP2D(self.config)
        self.include_layer_norm = include_layer_norm
        if self.include_layer_norm:
            self.ln_f = LayerNormDistributed(self.rg_tp2)

    def build(self, input_ids: popxl.Tensor):
        x = self.embedding(input_ids)
        x = self.decoder(x)
        if self.include_layer_norm:
            x = self.ln_f(x)
        return x

    @staticmethod
    def hf_mapping(config: BloomConfig, variables: NamedTensors, hf_model: HFModel, layer_norm=True) -> WeightsDict:
        dtype = config.model.dtype
        tp2 = config.execution.tensor_parallel_2

        weights = WeightsDict()

        if layer_norm:
            weights.update(
                {
                    variables.ln_f.weight: shard(to_numpy(hf_model.ln_f.weight.data, dtype), tp2, 0),
                    variables.ln_f.bias: shard(to_numpy(hf_model.ln_f.bias.data, dtype), tp2, 0),
                }
            )

        weights.update(BloomEmbeddingTP2D.hf_mapping(config, variables.embedding, hf_model))

        for l in range(config.model.layers):
            weights.update(BloomDecoderBlockTP2D.hf_mapping(config, variables.decoder[l], hf_model.h[l]))

        return weights
