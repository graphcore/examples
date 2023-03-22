# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
from config import GPTJConfig
import torch

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.named_tensors import NamedTensorData

from popxl_addons.layers import LayerNorm

from .embedding import GPTJEmbeddingsTP
from .decoder import GPTJDecoderTP, GPTJDecoderBlockTP

from transformers.models.gptj.modeling_gptj import GPTJModel as HFModel
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTJConfigHF


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
    def hf_mapping(
        config: GPTJConfig, variables: NamedTensors, hf_model: HFModel, layer_norm=True
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {}
        if layer_norm:
            weights = {
                variables.ln_f.weight: to_numpy(hf_model.ln_f.weight.data, dtype),
                variables.ln_f.bias: to_numpy(hf_model.ln_f.bias.data, dtype),
            }

        weights.update(GPTJEmbeddingsTP.hf_mapping(config, variables.embeddings, hf_model))

        for l in range(config.model.layers):
            weights.update(GPTJDecoderBlockTP.hf_mapping(config, variables.decoder[l], hf_model.h[l]))

        return weights

    @staticmethod
    def to_hf(variables_data: NamedTensorData, hf_model: HFModel, layer_norm=True) -> Dict[str, torch.Tensor]:
        state_dict = {}
        if layer_norm:
            state_dict["ln_f.weight"] = torch.tensor(variables_data.ln_f.weight, dtype=hf_model.config.torch_dtype)
            state_dict["ln_f.bias"] = torch.tensor(variables_data.ln_f.bias, dtype=hf_model.config.torch_dtype)

        state_dict.update(GPTJEmbeddingsTP.to_hf(hf_model.config, variables_data.embeddings, hf_model.wte))
        for l in range(hf_model.config.n_layer):
            state_dict.update(
                {
                    "h." + str(l) + "." + k: v
                    for k, v in GPTJDecoderBlockTP.to_hf(
                        hf_model.config, variables_data.decoder[l], hf_model.h[l]
                    ).items()
                }
            )
        return state_dict
