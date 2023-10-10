# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Dict
import torch
import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.t5.modeling_t5 import T5LayerNorm as HFModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF

import popxl_addons as addons
from config import T5Config
from popxl_addons.named_tensors import NamedTensorData
import numpy as np

from popxl_addons.named_tensors import NamedTensors


class T5LayerNorm(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.eps = config.model.eps
        self.dtype = config.model.dtype

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """
        Build layer normalisation for T5. No bias and no subtraction of mean.
        """
        w = self.add_variable_input("weight", partial(np.ones, x.shape[-1]), self.dtype)

        # Perform the computation in float32
        if x.dtype == popxl.float16:
            x = ops.cast(x, popxl.float32)
        variance = ops.mean(x * x, -1, keepdims=True)
        x = x / ops.sqrt(variance + self.eps)

        # Cast back down to float16 if needed
        if w.dtype == popxl.float16:
            x = ops.cast(x, popxl.float16)

        x = x * w
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.weight: to_numpy(hf_model.weight.data, dtype),
        }
        return weights

    @staticmethod
    def to_hf(config: T5ConfigHF, popxl_state_dict: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        state_dict = {}
        state_dict["weight"] = torch.tensor(popxl_state_dict.weight, dtype=config.torch_dtype)
        return state_dict
