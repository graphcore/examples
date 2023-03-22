# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import math
from functools import partial
from itertools import cycle

import numpy as np
import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import shard, shard2D
from popxl_addons.layers import Linear
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce
from popxl_addons.utils import WeightsDict
from transformers.models.bloom.modeling_bloom import BloomBlock

import popxl
from config import BloomConfig
from modelling.alibi import build_alibi_data
from popxl import ops
from popxl.utils import to_numpy
from utils.utils import tp2d_replica_groups


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    head_hidden_size = x.shape[1] // heads
    return x.reshape((1, sequence_length, heads, head_hidden_size))


def transpose_for_scores(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
    assert len(x.shape) == 4
    perm = (0, 2, 1, 3) if not is_key else (0, 2, 3, 1)
    return x.transpose(perm)


class BloomSelfAttentionTP2D(addons.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)
        self.tp1, self.tp2 = (
            config.execution.tensor_parallel_1,
            config.execution.tensor_parallel_2,
        )

        n_heads_groups = self.rg_tp1.group_size

        assert self.config.model.attention.heads % n_heads_groups == 0

        self.n_heads = self.config.model.attention.heads // n_heads_groups
        self.n_heads_all = self.config.model.attention.heads
        self.seq_len = self.config.model.sequence_length
        self.dtype = self.config.model.dtype

        self.ln_1 = LayerNormDistributed(self.rg_tp2)
        self.qkv = Linear(
            3 * self.config.model.hidden_size // n_heads_groups,
            bias=False,
            replica_grouping=self.rg_tp_all.transpose(),
        )

        self.output = Linear(
            self.config.model.hidden_size // self.rg_tp2.group_size,
            bias=False,
            replica_grouping=self.rg_tp_all.transpose(),
        )

    @staticmethod
    def _build_alibi_tensor(seq_len: int, heads: int, dtype, tp1: int, tp2: int) -> np.array:
        alibi = build_alibi_data(seq_len, heads, dtype)
        alibi = shard(alibi, tp1, 0)
        return alibi

    def _build_alibi_iter(self):
        # The alibi tensor is constant across one tp axis, but changes across
        # another. This type of "constant" is not supported in `popxl.constant`,
        # which must be the same across all replicas.A `_build_alibi_tensor`
        # will shard the tensor over `tp1` which `cycle` then uses to build an
        # iterator of length `tp1`. When the iterator is exhausted, it restarts
        # iteration from index 0. This will produce the desired effect of `tp1`
        # having different data, but identical data across `tp2`.
        return cycle(
            BloomSelfAttentionTP2D._build_alibi_tensor(
                self.seq_len,
                self.n_heads_all,
                self.dtype.as_numpy(),
                self.tp1,
                self.tp2,
            )
        )

    @popxl.in_sequence(True)
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        z = self.ln_1(x)
        z = self.qkv(z)
        z = replicated_all_reduce(z, group=self.rg_tp2)

        self.qkv_bias = self.add_variable_input(
            "qkv_bias",
            partial(np.zeros, z.shape[-1]),
            z.dtype,
            replica_grouping=self.rg_tp2,
        )
        z = z + self.qkv_bias

        query, key, value = ops.split(z, 3, axis=-1)

        def transform_heads(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
            return transpose_for_scores(reshape_for_scores(x, self.seq_len, self.n_heads), is_key)

        query = transform_heads(query, is_key=False)
        key = transform_heads(key, is_key=True)
        value = transform_heads(value, is_key=False)

        self.causal_mask = self.add_variable_input(
            "causal_mask",
            cycle(
                1e4
                * (np.tril(np.ones((self.seq_len, self.seq_len), dtype=query.dtype.as_numpy())) - 1)[np.newaxis, ...]
            ),
            query.dtype,
            replica_grouping=self.rg_tp_all,
        )

        # Using alibi attention bias rather than positional embeddings
        self.alibi = self.add_variable_input("alibi", self._build_alibi_iter(), x.dtype, replica_grouping=self.rg_tp2)

        attn_weights = query @ key
        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))
        ops.add_(attn_weights, self.alibi)
        ops.add_(attn_weights, self.causal_mask)

        attn_scores = ops.softmax(attn_weights, axis=-1)

        attn_output = attn_scores @ value
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape((x.shape[0], -1))

        z = self.output(attn_output)
        z = replicated_all_reduce(z, group=self.rg_tp1)
        self.output_bias = self.add_variable_input(
            "output_bias",
            lambda: np.zeros(z.shape[-1]),
            z.dtype,
            replica_grouping=self.rg_tp2.transpose(),
        )

        z = z + self.output_bias
        z = x + z

        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: BloomBlock):
        tp1 = config.execution.tensor_parallel_1
        tp2 = config.execution.tensor_parallel_2
        hidden_dim = config.model.hidden_size
        heads = config.model.attention.heads
        dtype = config.model.dtype
        sequence_length = config.model.sequence_length

        qkv_w = hf_model.self_attention.query_key_value.weight.data.T.reshape(hidden_dim, heads, 3, -1)
        hf_query_w, hf_key_w, hf_value_w = np.split(qkv_w, 3, axis=-2)

        hf_query_w = to_numpy(hf_query_w, dtype).reshape(hidden_dim, hidden_dim)
        hf_key_w = to_numpy(hf_key_w, dtype).reshape(hidden_dim, hidden_dim)
        hf_value_w = to_numpy(hf_value_w, dtype).reshape(hidden_dim, hidden_dim)

        qkv_b = hf_model.self_attention.query_key_value.bias.data.reshape(heads, 3, -1)
        hf_query_b, hf_key_b, hf_value_b = np.split(qkv_b, 3, axis=1)

        hf_query_b = to_numpy(hf_query_b, dtype).reshape(-1)
        hf_key_b = to_numpy(hf_key_b, dtype).reshape(-1)
        hf_value_b = to_numpy(hf_value_b, dtype).reshape(-1)

        query_w = shard2D(to_numpy(hf_query_w, dtype), tp1, tp2, 1, 0)
        key_w = shard2D(to_numpy(hf_key_w, dtype), tp1, tp2, 1, 0)
        value_w = shard2D(to_numpy(hf_value_w, dtype), tp1, tp2, 1, 0)

        query_b = np.split(to_numpy(hf_query_b, dtype), tp2, axis=0)
        key_b = np.split(to_numpy(hf_key_b, dtype), tp2, axis=0)
        value_b = np.split(to_numpy(hf_value_b, dtype), tp2, axis=0)

        qkv_weight = np.concatenate((query_w, key_w, value_w), axis=2)
        qkv_bias = np.concatenate((query_b, key_b, value_b), axis=1)

        output_w = shard2D(to_numpy(hf_model.self_attention.dense.weight.data.T, dtype), tp1, tp2, 0, 1)
        output_b = shard(to_numpy(hf_model.self_attention.dense.bias.data, dtype), tp2, 0)

        causal_mask = 1e4 * (np.tril(np.ones((sequence_length, sequence_length), dtype=dtype.as_numpy())) - 1)

        return WeightsDict(
            {
                variables.ln_1.weight: shard(to_numpy(hf_model.input_layernorm.weight.data, dtype), tp2, 0),
                variables.ln_1.bias: shard(to_numpy(hf_model.input_layernorm.bias.data, dtype), tp2, 0),
                variables.qkv.weight: qkv_weight,
                variables.qkv_bias: qkv_bias,
                variables.output.weight: output_w,
                variables.output_bias: output_b,
                variables.alibi: BloomSelfAttentionTP2D._build_alibi_tensor(
                    config.model.sequence_length, heads, dtype.as_numpy(), tp1, tp2
                ),
                variables.causal_mask: causal_mask,
            }
        )
