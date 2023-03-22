# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Optional, List, Dict

from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput

import popxl
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from config import BertConfig
from popxl_addons.layers import Linear, LayerNorm
import numpy as np

from popxl_addons import NamedTensors


class FeedForward(addons.Module):
    def __init__(self, config: BertConfig, ff_size: Optional[int] = None):
        super().__init__()
        self.config = config
        # Also known as the intermediate size
        self.ff_size = 4 * config.model.hidden_size if ff_size is None else ff_size

        # Layers
        self.intermediate = Linear(self.ff_size)
        self.output = Linear(config.model.hidden_size)
        self.norm = LayerNorm()

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        y = self.intermediate(x)
        y = ops.gelu(y)
        y = self.output(y)

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            y = ops.dropout(y, seed, p=self.config.model.dropout_prob)
        y = self.norm(y + x)

        return y

    @staticmethod
    def hf_mapping(
        config: BertConfig, variables: NamedTensors, hf_model_int: BertIntermediate, hf_model_out: BertOutput
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        return {
            variables.intermediate.weight: np.ascontiguousarray(to_numpy(hf_model_int.dense.weight.T, dtype)),
            variables.intermediate.bias: to_numpy(hf_model_int.dense.bias, dtype),
            variables.output.weight: np.ascontiguousarray(to_numpy(hf_model_out.dense.weight.T, dtype)),
            variables.output.bias: to_numpy(hf_model_out.dense.bias, dtype),
            variables.norm.weight: to_numpy(hf_model_out.LayerNorm.weight, dtype),
            variables.norm.bias: to_numpy(hf_model_out.LayerNorm.bias, dtype),
        }
