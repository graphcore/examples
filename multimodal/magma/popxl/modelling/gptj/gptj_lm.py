# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
from typing import Optional, Dict, Tuple, Callable, List, Union
import torch

# HF
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM as HFModel
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTJConfigHF

import popxl
from popxl.utils import to_numpy
from popxl import ops

import popxl_addons as addons
from popxl_addons import NamedTensors, GraphWithNamedArgs
from popxl_addons.layers import Linear
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_inputs
from popxl_addons.layers import LayerNorm
from popxl_addons.named_tensors import NamedTensorData

from modelling.gptj.gptj_model import GPTJModelTP
from configs import GPTJConfig
from popxl_addons.array_munging import shard
from math import ceil


class GPTJLMHeadTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        """
        Language model head for GPTJ, with weights sharded along the vocab axis.
        Includes a layer norm which is normally after the decoder stack. Included here for phased execution.
        Outputs sharded logits through a linear projection.
        """
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=1)
        # identical
        self.ln_f = LayerNorm()
        shard_size = ceil(self.config.embedding.vocab_size / tp)
        self.head = Linear(shard_size, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x = self.ln_f(x)
        # sharded
        x = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())
        logits = self.head(x)
        return logits

    @staticmethod
    def finetuneanon_mapping(
        config: GPTJConfig, variables: NamedTensors, hf_model: HFModel
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype
        n_shards = config.execution.tensor_parallel

        weights = {
            variables.head.weight: shard(to_numpy(hf_model.lm_head.weight.data.T, dtype), n_shards, axis=-1),
            variables.head.bias: shard(to_numpy(hf_model.lm_head.bias.data, dtype), n_shards, axis=-1),
            variables.ln_f.weight: to_numpy(hf_model.transformer.ln_f.weight.data, dtype),
            variables.ln_f.bias: to_numpy(hf_model.transformer.ln_f.bias.data, dtype),
        }

        return weights


class GPTJLMHeadModelTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        """GPTJ model with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = GPTJModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.lm_head = GPTJLMHeadTP(config)

    def build(self, input_ids: popxl.Tensor) -> popxl.Tensor:

        x = self.transformer(input_ids)
        x = self.lm_head(x)

        return x

    @staticmethod
    def finetuneanon_mapping(
        config: GPTJConfig, variables: NamedTensors, hf_model: HFModel, from_magma: bool = True
    ) -> Dict[popxl.Tensor, np.ndarray]:
        weights = GPTJModelTP.finetuneanon_mapping(
            config, variables.transformer, hf_model.transformer, layer_norm=False, from_magma=from_magma
        )
        weights.update(GPTJLMHeadTP.finetuneanon_mapping(config, variables.lm_head, hf_model))

        return weights
