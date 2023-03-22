# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Optional
from typing import Dict

import numpy as np
from transformers.models.bert.modeling_bert import BertPreTrainingHeads as HFBertPreTrainingHeads

import popxl
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from config import BertConfig
from popxl_addons import NamedTensors
from popxl_addons.layers import Embedding, Linear, LayerNorm


class BertMLM(addons.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.transform = Linear(self.config.model.hidden_size)
        self.norm = LayerNorm()

    def build(
        self, x: popxl.Tensor, word_embedding_t: popxl.Tensor, masked_positions: Optional[popxl.Tensor] = None
    ) -> popxl.Tensor:

        if masked_positions is not None:
            masked_positions = masked_positions.reshape((-1, self.config.model.mlm.mask_tokens))
            offset = np.arange(0, self.config.execution.micro_batch_size) * self.config.model.sequence_length
            masked_positions = masked_positions + offset.reshape(-1, 1)
            x = ops.gather(x.reshape((-1, self.config.model.hidden_size)), masked_positions.flatten_(), axis=0)

        x = self.transform(x)
        x = ops.gelu(x)
        x = self.norm(x)

        b = self.add_variable_input("bias", partial(np.zeros, word_embedding_t.shape[1]), x.dtype)
        logits = (x @ word_embedding_t) + b

        return logits

    @staticmethod
    def hf_mapping(
        config: BertConfig, variables: NamedTensors, hf_model: HFBertPreTrainingHeads
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        return {
            variables.transform.weight: to_numpy(hf_model.predictions.transform.dense.weight.data.T, dtype),
            variables.transform.bias: to_numpy(hf_model.predictions.transform.dense.bias.data, dtype),
            variables.norm.weight: to_numpy(hf_model.predictions.transform.LayerNorm.weight.data, dtype),
            variables.norm.bias: to_numpy(hf_model.predictions.transform.LayerNorm.bias.data, dtype),
            variables.bias: to_numpy(hf_model.predictions.decoder.bias.data, dtype),
        }
