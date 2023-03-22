# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
from configs import GPTJConfig
import torch

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.named_tensors import NamedTensorData

from popxl_addons.layers import LayerNorm

from .embedding import GPTJEmbeddingsTP
from .decoder import GPTJDecoderTP, GPTJDecoderBlockTP

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoModel as HFModel
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTJConfigHF


class GPTJModelTP(addons.Module):
    def __init__(self, config: GPTJConfig, include_layer_norm=True):
        super().__init__()
        self.config = config
        # sharded, then last bit identical
        self.embeddings = GPTJEmbeddingsTP(self.config)
        # identical inputs, then sharded, then identical
        self.decoder = GPTJDecoderTP(self.config)
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
    def finetuneanon_mapping(
        config: GPTJConfig, variables: NamedTensors, hf_model: HFModel, layer_norm=True, from_magma: bool = True
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype
        weights = {}
        if layer_norm:
            weights = {
                variables.ln_f.weight: to_numpy(hf_model.ln_f.weight.data, dtype),
                variables.ln_f.bias: to_numpy(hf_model.ln_f.bias.data, dtype),
            }

        weights.update(GPTJEmbeddingsTP.finetuneanon_mapping(config, variables.embeddings, hf_model))

        for l in range(config.layers):
            weights.update(
                GPTJDecoderBlockTP.finetuneanon_mapping(
                    config, variables.decoder[l], hf_model.h[l], from_magma=from_magma
                )
            )

        return weights
