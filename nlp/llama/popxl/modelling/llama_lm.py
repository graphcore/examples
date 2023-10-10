# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Optional, Dict, Tuple, Callable, List, Union

# HF
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFModel

import popxl
from popxl.utils import to_numpy
from popxl import ops

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.layers import Linear
from .rms_norm import LlamaRMSNorm

from modelling.llama_model import LlamaModelTP
from config import LlamaConfig
from popxl_addons.array_munging import shard
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce

from math import ceil


def gather_logits_tp(config: LlamaConfig, logits: popxl.Tensor, last_token_index: popxl.Tensor):
    tp = config.execution.tensor_parallel

    if not config.execution.use_cache:
        offset_last_token_index = last_token_index + popxl.constant(
            np.asarray([i * config.model.sequence_length for i in range(config.execution.micro_batch_size)]),
            dtype=popxl.int32,
        )

        next_token_logits = logits[offset_last_token_index]  # (tp, mb_size, vocab_shard_size)
    else:
        next_token_logits = logits

    next_token_logits = ops.collectives.replicated_all_gather(
        next_token_logits, group=popxl.gcg().ir.replica_grouping(group_size=tp), output_shape="new_axis"
    )
    next_token_logits = next_token_logits.transpose((1, 0, 2)).reshape_((config.execution.micro_batch_size, -1))

    next_token_logits = next_token_logits[:, : config.model.embedding.vocab_size]

    return next_token_logits


class LlamaLMHeadTP(addons.Module):
    def __init__(self, config: LlamaConfig):
        """
        Language model head for Llama, with weights sharded along the vocab axis.
        Includes a layer norm which is normally after the decoder stack. Included here for phased execution.
        Outputs sharded logits through a linear projection.
        """
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # identical
        shard_size = ceil(self.config.model.embedding.vocab_size / tp)

        self.ln_f = LlamaRMSNorm(self.config)
        self.head = Linear(shard_size, replica_grouping=self.replica_grouping, bias=False)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @staticmethod
    def hf_mapping(config: LlamaConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        def pad(x, n_pad):
            return np.pad(x, ((0, n_pad), (0, 0)))

        shard_size = ceil(config.model.embedding.vocab_size / n_shards)
        num_pad = shard_size * n_shards - config.model.embedding.vocab_size

        weights = {
            variables.head.weight: shard(
                pad(to_numpy(hf_model.lm_head.weight.data, dtype), num_pad).T, n_shards, axis=-1
            ),
            variables.ln_f.weight: to_numpy(hf_model.model.norm.weight.data, dtype),
        }

        return weights


class LlamaLMHeadModelTP(addons.Module):
    def __init__(self, config: LlamaConfig):
        """Llama model with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = LlamaModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.lm_head = LlamaLMHeadTP(config)

    def build(self, input_ids: popxl.Tensor) -> popxl.Tensor:
        x = self.transformer(input_ids)
        x = self.lm_head(x)

        return x

    @staticmethod
    def hf_mapping(config: LlamaConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        weights = LlamaModelTP.hf_mapping(config, variables.transformer, hf_model.model, layer_norm=False)
        weights.update(LlamaLMHeadTP.hf_mapping(config, variables.lm_head, hf_model))

        return weights
