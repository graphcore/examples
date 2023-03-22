# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
import math

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from utils.utils import shard
from popxl_addons.layers import Linear, LayerNorm

from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)

from config import GPTConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    assert len(x.shape) == 2
    micro_batch_size = x.shape[0] // sequence_length
    head_hidden_size = x.shape[1] // heads
    return x.reshape((micro_batch_size, sequence_length, heads, head_hidden_size))


def transpose_for_scores(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
    assert len(x.shape) == 4
    perm = (0, 2, 1, 3) if not is_key else (0, 2, 3, 1)
    return x.transpose(perm)


class GPTAttentionHeads(addons.Module):
    def __init__(self, config: GPTConfig, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.config = config
        self.replica_grouping = replica_grouping

        if self.replica_grouping:
            n_heads_groups = self.replica_grouping.num_groups
        else:
            n_heads_groups = 1

        assert self.config.model.attention.heads % n_heads_groups == 0

        self.n_heads_groups = n_heads_groups
        self.n_heads = self.config.model.attention.heads // n_heads_groups
        self.qkv = Linear(3 * self.config.model.hidden_size // n_heads_groups, replica_grouping=replica_grouping)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None):
        qkv_act = self.qkv(x)
        query, key, value = ops.split(qkv_act, 3, axis=-1)

        def transform_heads(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
            return transpose_for_scores(reshape_for_scores(x, self.config.model.sequence_length, self.n_heads), is_key)

        query = transform_heads(query, False)
        key = transform_heads(key, is_key=True)
        value = transform_heads(value, False)

        attn_weights = query @ key
        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))

        query_length, key_length = attn_weights.shape[-2], attn_weights.shape[-1]
        # TODO: This causes a large rearrangement. Check tile mapping of broadcast add
        causal_mask = popxl.constant(
            np.tril(np.ones((query_length, key_length))), attn_weights.dtype, name="causal_mask"
        )

        attn_weights = attn_weights + 1e4 * (causal_mask - 1)
        attn_scores = ops.softmax(attn_weights, axis=-1)
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            attn_scores = ops.dropout(attn_scores, seed, p=self.config.model.dropout_prob)

        attn_output = attn_scores @ value

        x_part_shape = list(x.shape)
        x_part_shape[-1] = x_part_shape[-1] // self.n_heads_groups
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(x_part_shape)

        return attn_output

    @staticmethod
    def hf_mapping(config: GPTConfig, vars: NamedTensors, hf_model: GPT2Attention) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        hf_query_w, hf_key_w, hf_value_w = np.split(hf_model.c_attn.weight.data.numpy(), 3, axis=-1)
        hf_query_b, hf_key_b, hf_value_b = np.split(hf_model.c_attn.bias.data.numpy(), 3, axis=-1)

        query_w = to_numpy(hf_query_w, dtype)
        key_w = to_numpy(hf_key_w, dtype)
        value_w = to_numpy(hf_value_w, dtype)
        query_b = to_numpy(hf_query_b, dtype)
        key_b = to_numpy(hf_key_b, dtype)
        value_b = to_numpy(hf_value_b, dtype)

        return {
            vars.qkv.weight: np.ascontiguousarray(np.concatenate((query_w, key_w, value_w), axis=-1)),
            vars.qkv.bias: np.concatenate((query_b, key_b, value_b), axis=-1),
        }


class GPTSelfAttentionTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

        # Identical across devices
        self.ln_1 = LayerNorm()

        # Sharded across devices
        self.heads = GPTAttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices (bias applied separately)
        self.output = Linear(self.config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""
        heads_seed = None
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            seed, heads_seed = ops.split_random_seed(seed)

        # ----- Identical computation -----

        z = self.ln_1(x)

        z = replicated_all_reduce_identical_inputs(z, group=self.replica_grouping.transpose())
        # ----- Sharded computation -----
        z = self.heads(z, seed=heads_seed)
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        # ----- Identical computation -----
        # Output linear layer bias (identical bias on all devices)
        self.bias = self.add_variable_input("bias", lambda: np.zeros(z.shape[-1]), z.dtype)
        z = z + self.bias

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        z = x + z

        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: GPT2Block) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        hf_query_w, hf_key_w, hf_value_w = np.split(hf_model.attn.c_attn.weight.data.numpy(), 3, axis=-1)
        hf_query_b, hf_key_b, hf_value_b = np.split(hf_model.attn.c_attn.bias.data.numpy(), 3, axis=-1)

        query_w = np.split(to_numpy(hf_query_w, dtype), n_shards, axis=-1)
        key_w = np.split(to_numpy(hf_key_w, dtype), n_shards, axis=-1)
        value_w = np.split(to_numpy(hf_value_w, dtype), n_shards, axis=-1)
        query_b = np.split(to_numpy(hf_query_b, dtype), n_shards, axis=-1)
        key_b = np.split(to_numpy(hf_key_b, dtype), n_shards, axis=-1)
        value_b = np.split(to_numpy(hf_value_b, dtype), n_shards, axis=-1)
        c_proj_w = to_numpy(hf_model.attn.c_proj.weight.data.numpy(), dtype)
        c_proj_b = to_numpy(hf_model.attn.c_proj.bias.data, dtype)

        return {
            variables.ln_1.weight: to_numpy(hf_model.ln_1.weight.data, dtype),
            variables.ln_1.bias: to_numpy(hf_model.ln_1.bias.data, dtype),
            variables.heads.qkv.weight: np.ascontiguousarray(
                np.concatenate(
                    [
                        np.concatenate([query_w[i], key_w[i], value_w[i]], axis=-1)[np.newaxis, ...]
                        for i in range(n_shards)
                    ]
                )
            ),
            variables.heads.qkv.bias: np.ascontiguousarray(
                np.concatenate(
                    [
                        np.concatenate([query_b[i], key_b[i], value_b[i]], axis=-1)[np.newaxis, ...]
                        for i in range(n_shards)
                    ]
                )
            ),
            variables.output.weight: shard(c_proj_w, n_shards, axis=0),
            variables.bias: c_proj_b,  # Copied
        }
