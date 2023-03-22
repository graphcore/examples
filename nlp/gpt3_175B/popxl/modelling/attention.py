# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import numpy as np
import math

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from utils.utils import tp2d_replica_groups
from popxl_addons.utils import WeightsDict
from popxl_addons.array_munging import shard, shard2D
from popxl_addons.layers import Linear, LayerNorm

from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)

from config import GPTConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


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

        assert (
            self.config.model.attention.heads % n_heads_groups == 0
        ), f"{self.config.model.attention.heads} % {n_heads_groups} != 0"

        self.n_heads_groups = n_heads_groups
        self.n_heads = self.config.model.attention.heads // n_heads_groups
        self.qkv = Linear(3 * self.config.model.hidden_size // n_heads_groups, replica_grouping=replica_grouping)

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
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
        causal_mask = popxl.constant(
            # HF uses 1e9 which is beyond fp16 range
            1e4 * (np.tril(np.ones((self.config.model.sequence_length, self.config.model.sequence_length))) - 1),
            query.dtype,
            name="causal_mask",
        )

        if self.config.execution.attention_serialisation > 1:
            queries = ops.split(query, self.config.execution.attention_serialisation, axis=2)
            masks = ops.split(causal_mask, self.config.execution.attention_serialisation, axis=0)

            blk_graph = popxl.gcg().ir.create_graph(self.attention_block, queries[0], key, value, masks[0], seed)

            attn_outputs = []
            for query_i, mask_i in zip(queries, masks):
                args = [query_i, key, value, mask_i]
                # Each step should have different dropout
                if seed is not None:
                    seed, blk_seed = ops.split_random_seed(seed)
                    args.append(blk_seed)

                (attn_block_output,) = ops.call(blk_graph, *args)

                attn_outputs.append(attn_block_output)
            attn_output = ops.concat(attn_outputs, axis=2)
        else:
            attn_output = self.attention_block(query, key, value, causal_mask, seed)

        return attn_output.transpose((0, 2, 1, 3)).reshape(
            (self.config.execution.micro_batch_size * self.config.model.sequence_length, -1)
        )

    def attention_block(
        self, query: popxl.Tensor, key: popxl.Tensor, value: popxl.Tensor, mask: popxl.Tensor, seed: popxl.Tensor
    ):
        attn_weights = query @ key

        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))
        attn_weights = attn_weights + mask

        attn_scores = ops.softmax(attn_weights, axis=-1)
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            attn_scores = ops.dropout(attn_scores, seed, p=self.config.model.dropout_prob)

        return attn_scores @ value


class GPTSelfAttentionTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        tp = config.execution.tensor_parallel_1
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
    def hf_mapping(config, variables: NamedTensors, hf_model: GPT2Block) -> WeightsDict:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel_1

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

        return WeightsDict(
            {
                **LayerNorm.torch_mapping(variables.ln_1, hf_model.ln_1, dtype),
                variables.heads.qkv.weight: np.concatenate(
                    [
                        np.concatenate([query_w[i], key_w[i], value_w[i]], axis=-1)[np.newaxis, ...]
                        for i in range(n_shards)
                    ]
                ),
                variables.heads.qkv.bias: np.concatenate(
                    [
                        np.concatenate([query_b[i], key_b[i], value_b[i]], axis=-1)[np.newaxis, ...]
                        for i in range(n_shards)
                    ]
                ),
                variables.output.weight: shard(c_proj_w, n_shards, axis=0),
                variables.bias: c_proj_b,  # Copied
            }
        )


class GPTAttentionHeadsTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        n_heads_groups = self.rg_tp1.group_size

        assert self.config.model.attention.heads % n_heads_groups == 0

        self.n_heads_groups = n_heads_groups
        self.n_heads = self.config.model.attention.heads // n_heads_groups
        self.qkv = Linear(
            3 * self.config.model.hidden_size // n_heads_groups, bias=False, replica_grouping=self.rg_tp_all.transpose()
        )

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        # x: [b*s, h/tp2]
        # input data: identical tp1, sharded tp2
        # computation: sharded tp1, sharded tp2
        z = self.qkv(x)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.rg_tp2)
        # z: sharded tp1, identical tp2

        self.qkv_bias = self.add_variable_input(
            "qkv_bias", partial(np.zeros, z.shape[-1]), z.dtype, replica_grouping=self.rg_tp2.transpose()
        )
        z = z + self.qkv_bias

        query, key, value = ops.split(z, 3, axis=-1)

        def transform_heads(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
            return transpose_for_scores(reshape_for_scores(x, self.config.model.sequence_length, self.n_heads), is_key)

        query = transform_heads(query, False)
        key = transform_heads(key, is_key=True)
        value = transform_heads(value, False)

        causal_mask = popxl.constant(
            # HF uses 1e9 which is beyond fp16 range
            1e4 * (np.tril(np.ones((self.config.model.sequence_length, self.config.model.sequence_length))) - 1),
            query.dtype,
            name="causal_mask",
        )

        if self.config.execution.attention_serialisation > 1:
            queries = ops.split(query, self.config.execution.attention_serialisation, axis=2)
            masks = ops.split(causal_mask, self.config.execution.attention_serialisation, axis=0)

            blk_graph = popxl.gcg().ir.create_graph(self.attention_block, queries[0], key, value, masks[0], seed)

            attn_outputs = []
            for query_i, mask_i in zip(queries, masks):
                args = [query_i, key, value, mask_i]
                # Each step should have different dropout
                if seed is not None:
                    seed, blk_seed = ops.split_random_seed(seed)
                    args.append(blk_seed)

                (attn_block_output,) = ops.call(blk_graph, *args)

                attn_outputs.append(attn_block_output)
            attn_output = ops.concat(attn_outputs, axis=2)
        else:
            attn_output = self.attention_block(query, key, value, causal_mask, seed)

        attn_output = replicated_all_reduce_identical_inputs(attn_output, group=self.rg_tp2)

        # z: sharded tp1, identical tp2
        return attn_output.transpose((0, 2, 1, 3)).reshape(
            (self.config.execution.micro_batch_size * self.config.model.sequence_length, -1)
        )

    def attention_block(
        self, query: popxl.Tensor, key: popxl.Tensor, value: popxl.Tensor, mask: popxl.Tensor, seed: popxl.Tensor
    ):
        attn_weights = query @ key

        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))
        attn_weights = attn_weights + mask
        attn_scores = ops.softmax(attn_weights, axis=-1)

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            attn_scores = ops.dropout(attn_scores, seed, p=self.config.model.dropout_prob)

        attn_output = attn_scores @ value

        return attn_output


class GPTSelfAttentionTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        self.ln_1 = LayerNormDistributed(self.rg_tp2)

        self.heads = GPTAttentionHeadsTP2D(config=config)

        self.output = Linear(
            self.config.model.hidden_size // self.rg_tp2.group_size,
            bias=False,
            replica_grouping=self.rg_tp_all.transpose(),
        )

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""
        # x: [b*s, h/tp2]
        # x: identical tp1, sharded tp2

        heads_seed = None
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            seed, heads_seed = ops.split_random_seed(seed)

        z = self.ln_1(x)

        z = replicated_all_reduce_identical_inputs(z, group=self.rg_tp1)
        # z: identical tp1, sharded tp2

        z = self.heads(z, seed=heads_seed)
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.rg_tp1)
        # z: identical tp1, sharded tp2

        self.output_bias = self.add_variable_input(
            "output_bias", lambda: np.zeros(z.shape[-1]), z.dtype, replica_grouping=self.rg_tp2.transpose()
        )
        z = z + self.output_bias

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        z = x + z

        # z: identical tp1, sharded tp2
        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: GPT2Block) -> WeightsDict:
        dtype = config.model.dtype
        tp1 = config.execution.tensor_parallel_1
        tp2 = config.execution.tensor_parallel_2

        hf_query_w, hf_key_w, hf_value_w = np.split(hf_model.attn.c_attn.weight.data.numpy(), 3, axis=1)
        hf_query_b, hf_key_b, hf_value_b = np.split(hf_model.attn.c_attn.bias.data.numpy(), 3, axis=0)

        query_w = shard2D(to_numpy(hf_query_w, dtype), tp1, tp2, 1, 0)
        key_w = shard2D(to_numpy(hf_key_w, dtype), tp1, tp2, 1, 0)
        value_w = shard2D(to_numpy(hf_value_w, dtype), tp1, tp2, 1, 0)

        query_b = np.split(to_numpy(hf_query_b, dtype), tp2, axis=0)
        key_b = np.split(to_numpy(hf_key_b, dtype), tp2, axis=0)
        value_b = np.split(to_numpy(hf_value_b, dtype), tp2, axis=0)

        output_w = shard2D(to_numpy(hf_model.attn.c_proj.weight.data.numpy(), dtype), tp1, tp2, 0, 1)
        output_b = shard(to_numpy(hf_model.attn.c_proj.bias.data, dtype), tp2, 0)

        return WeightsDict(
            {
                variables.ln_1.weight: shard(to_numpy(hf_model.ln_1.weight.data, dtype), tp2, 0),
                variables.ln_1.bias: shard(to_numpy(hf_model.ln_1.bias.data, dtype), tp2, 0),
                variables.heads.qkv.weight: np.concatenate((query_w, key_w, value_w), axis=2),
                variables.heads.qkv_bias: np.concatenate((query_b, key_b, value_b), axis=1),
                variables.output.weight: output_w,
                variables.output_bias: output_b,
            }
        )
