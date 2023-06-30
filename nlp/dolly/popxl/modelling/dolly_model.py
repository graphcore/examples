# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
from config import DollyConfig

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors

from popxl_addons.layers import LayerNorm

from .embedding import DollyEmbeddingsTP
from .decoder import DollyDecoderTP, DollyDecoderBlockTP

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel as HFModel


class DollyModelTP(addons.Module):
    def __init__(self, config: DollyConfig, include_layer_norm=True):
        super().__init__()
        self.config = config
        # sharded, then last bit identical
        self.embeddings = DollyEmbeddingsTP(self.config)
        # identical inputs, then sharded, then identical
        self.decoder = DollyDecoderTP(self.config)
        # identical
        self.include_layer_norm = include_layer_norm
        if self.include_layer_norm:
            self.ln_f = LayerNorm()

    def build(self, input_ids: popxl.Tensor):
        x = self.embeddings(input_ids)
        x = self.decoder(x)
        if self.include_layer_norm:
            x = self.ln_f(x)
        return x

    @staticmethod
    def hf_mapping(
        config: DollyConfig, variables: NamedTensors, hf_model: HFModel, layer_norm=True
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {}
        if layer_norm:
            weights = {
                variables.ln_f.weight: to_numpy(hf_model.final_layer_norm.weight.data, dtype),
                variables.ln_f.bias: to_numpy(hf_model.final_layer_norm.bias.data, dtype),
            }

        weights.update(DollyEmbeddingsTP.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.model.layers):
            weights.update(DollyDecoderBlockTP.hf_mapping(config, variables.decoder[l], hf_model.layers[l]))

        return weights
