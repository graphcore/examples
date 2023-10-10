# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
from config import LlamaConfig

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors

# from popxl_addons.layers import LayerNorm
from .rms_norm import LlamaRMSNorm

from .embedding import LlamaEmbeddingsTP
from .decoder import LlamaDecoderTP, LlamaDecoderBlockTP

from transformers.models.llama.modeling_llama import LlamaModel as HFModel


class LlamaModelTP(addons.Module):
    def __init__(self, config: LlamaConfig, include_layer_norm=True):
        super().__init__()
        self.config = config
        # sharded, then last bit identical
        self.embeddings = LlamaEmbeddingsTP(self.config)
        # identical inputs, then sharded, then identical
        self.decoder = LlamaDecoderTP(self.config)
        # identical
        self.include_layer_norm = include_layer_norm
        if self.include_layer_norm:
            self.ln_f = LlamaRMSNorm(self.config)

    def build(self, input_ids: popxl.Tensor):
        x = self.embeddings(input_ids)
        x = self.decoder(x)
        if self.include_layer_norm:
            x = self.ln_f(x)
        return x

    @staticmethod
    def hf_mapping(
        config: LlamaConfig, variables: NamedTensors, hf_model: HFModel, layer_norm=True
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        weights = {}
        if layer_norm:
            weights = {
                variables.ln_f.weight: to_numpy(hf_model.norm.weight.data, dtype),
            }

        weights.update(LlamaEmbeddingsTP.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.model.layers):
            weights.update(LlamaDecoderBlockTP.hf_mapping(config, variables.decoder[l], hf_model.layers[l]))

        return weights
