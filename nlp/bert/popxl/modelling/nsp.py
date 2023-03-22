# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np

import popxl
from popxl import ops
from popxl.utils import to_numpy
from transformers.models.bert.modeling_bert import (
    BertPreTrainingHeads as HFBertPreTrainingHeads,
    BertPooler as HFBertPooler,
)
from typing import Dict

import popxl_addons as addons
from config import BertConfig
from popxl_addons import NamedTensors
from popxl_addons.layers import Linear


class BertPooler(addons.Module):
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dense = Linear(self.config.model.hidden_size)

    def build(self, x: popxl.Tensor):
        x = x.reshape((-1, self.config.model.sequence_length, self.config.model.hidden_size))
        first_token_tensor = x[:, 0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = ops.tanh(pooled_output)
        return pooled_output


class BertNSP(addons.Module):
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.pooler = BertPooler(self.config)
        self.seq_relationship = Linear(2)

    def build(self, x: popxl.Tensor):
        pooled_x = self.pooler(x)
        return self.seq_relationship(pooled_x)

    @staticmethod
    def hf_mapping(
        config: BertConfig, variables: NamedTensors, hf_model: HFBertPreTrainingHeads, hf_model_pooler: HFBertPooler
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        return {
            variables.pooler.dense.weight: to_numpy(hf_model_pooler.dense.weight.data.T, dtype),
            variables.pooler.dense.bias: to_numpy(hf_model_pooler.dense.bias.data, dtype),
            variables.seq_relationship.weight: to_numpy(hf_model.seq_relationship.weight.data.T, dtype),
            variables.seq_relationship.bias: to_numpy(hf_model.seq_relationship.bias.data, dtype),
        }
