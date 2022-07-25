# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
from typing import Optional, Dict
from config import GPTConfig
from modelling.decoder import GPTDecoderBlockTP, GPTDecoderBlock

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.layers import LayerNorm
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs)

from .embedding import GPTEmbeddings, GPTEmbeddingsTP
from .decoder import GPTDecoder, GPTDecoderTP

from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HFModel


class GPTModel(addons.Module):
    def __init__(self, config: GPTConfig, include_layer_norm=True):
        super().__init__()
        self.config = config
        self.embeddings = GPTEmbeddings(self.config)
        self.decoder = GPTDecoder(self.config)
        self.include_layer_norm = include_layer_norm
        if self.include_layer_norm:
            self.ln_f = LayerNorm()

    def build(self, input_ids: popxl.Tensor,
              position_ids: popxl.Tensor = None):
        x = self.embeddings(input_ids, position_ids)
        x = self.decoder(x)
        if self.include_layer_norm:
            x = self.ln_f(x)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig,
                   variables: NamedTensors,
                   hf_model: HFModel,
                   layer_norm=True) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {}
        if layer_norm:
            weights = {
                variables.ln_f.weight: to_numpy(hf_model.ln_f.weight.data,
                                                dtype),
                variables.ln_f.bias: to_numpy(hf_model.ln_f.bias.data, dtype),
            }

        weights.update(
            GPTEmbeddings.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.model.layers):
            weights.update(
                GPTDecoderBlock.hf_mapping(config, variables.decoder[l],
                                           hf_model.h[l]))

        return weights


class GPTModelTP(addons.Module):
    def __init__(self,
                 config: GPTConfig,
                 include_layer_norm=True):
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
    def hf_mapping(config: GPTConfig,
                   variables: NamedTensors,
                   hf_model: HFModel,
                   layer_norm=True) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {}
        if layer_norm:
            weights = {
                variables.ln_f.weight: to_numpy(hf_model.ln_f.weight.data,
                                                dtype),
                variables.ln_f.bias: to_numpy(hf_model.ln_f.bias.data, dtype),
            }

        weights.update(
            GPTEmbeddingsTP.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.model.layers):
            weights.update(
                GPTDecoderBlockTP.hf_mapping(config, variables.decoder[l],
                                             hf_model.h[l]))

        return weights
