# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Optional, Dict

from popxl.tensor import Variable

from config import GPTConfig
from modelling.decoder import GPTDecoderBlockTP, GPTDecoderTP2D, GPTDecoderBlockTP2D

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict
from popxl_addons.array_munging import shard
from popxl_addons.layers import LayerNorm
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from utils.utils import tp2d_replica_groups

from .embedding import GPTEmbeddingsTP, GPTEmbeddingsTP2D
from .decoder import GPTDecoderTP

from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HFModel


class GPTModelTP(addons.Module):
    def __init__(self, config: GPTConfig, include_layer_norm: bool = True):
        super().__init__()
        self.config = config
        # sharded, then last bit identical
        self.embeddings = GPTEmbeddingsTP(self.config)
        # identical inputs, then sharded, then identical
        self.decoder = GPTDecoderTP(self.config)
        # identical
        self.include_layer_norm = include_layer_norm
        if self.include_layer_norm:
            self.ln_f = LayerNorm()

    def build(self, input_ids: popxl.Tensor, position_ids: popxl.Tensor):
        x = self.embeddings(input_ids, position_ids)
        x = self.decoder(x)
        if self.include_layer_norm:
            x = self.ln_f(x)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel, layer_norm=True) -> WeightsDict:
        dtype = config.model.dtype
        weights = WeightsDict()
        if layer_norm:
            weights = {
                variables.ln_f.weight: to_numpy(hf_model.ln_f.weight.data, dtype),
                variables.ln_f.bias: to_numpy(hf_model.ln_f.bias.data, dtype),
            }

        weights.update(GPTEmbeddingsTP.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.model.layers):
            weights.update(GPTDecoderBlockTP.hf_mapping(config, variables.decoder[l], hf_model.h[l]))

        return weights


class GPTModelTP2D(addons.Module):
    def __init__(self, config: GPTConfig, include_layer_norm: bool = True):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        self.embeddings = GPTEmbeddingsTP2D(self.config)
        self.decoder = GPTDecoderTP2D(self.config)
        self.include_layer_norm = include_layer_norm
        if self.include_layer_norm:
            self.ln_f = LayerNormDistributed(self.rg_tp2)

    def build(self, input_ids: popxl.Tensor, position_ids: popxl.Tensor):
        x = self.embeddings(input_ids, position_ids)
        x = self.decoder(x)
        if self.include_layer_norm:
            x = self.ln_f(x)
        return x

    @staticmethod
    def hf_mapping(
        config: GPTConfig, variables: NamedTensors, hf_model: HFModel, layer_norm: bool = True
    ) -> WeightsDict:
        dtype = config.model.dtype
        tp2 = config.execution.tensor_parallel_2

        weights = WeightsDict()
        if layer_norm:
            weights = {
                variables.ln_f.weight: shard(to_numpy(hf_model.ln_f.weight.data, dtype), tp2, 0),
                variables.ln_f.bias: shard(to_numpy(hf_model.ln_f.bias.data, dtype), tp2, 0),
            }

        weights.update(GPTEmbeddingsTP2D.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.model.layers):
            weights.update(GPTDecoderBlockTP2D.hf_mapping(config, variables.decoder[l], hf_model.h[l]))

        return weights
