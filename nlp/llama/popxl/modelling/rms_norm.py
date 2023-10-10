# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Dict
import torch
import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFModel

import popxl_addons as addons
from config import LlamaConfig
from popxl_addons.named_tensors import NamedTensorData
import numpy as np

from popxl_addons.named_tensors import NamedTensors


class LlamaRMSNorm(addons.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.eps = config.model.eps

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """
        Build RMS layer normalisation for Llama. No bias and no subtraction of mean. This is equivalent to T5LayerNorm.
        """
        w = self.add_variable_input("weight", partial(np.ones, x.shape[-1]), x.dtype)

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
    def hf_mapping(config: LlamaConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.weight: to_numpy(hf_model.weight.data, dtype),
        }
        return weights
