# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
import math

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import shard, repeat_axis
from popxl_addons.layers import Linear

from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce

from .rotary_pos_embed import rotary_pos_embed, trig_table_constants

from config import DollyConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention as HFModel


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    assert len(x.shape) == 2
    micro_batch_size = x.shape[0] // sequence_length
    head_hidden_size = x.shape[1] // heads
    return x.reshape((micro_batch_size, sequence_length, heads, head_hidden_size))


class DollyAttentionHeads(addons.Module):
    def __init__(self, config: DollyConfig, replica_grouping: Optional[ReplicaGrouping] = None):
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
        self.rotary_dim = self.config.model.attention.rotary_dim or self.config.model.hidden_size // self.n_heads

    def build(self, x: popxl.Tensor):
        # x: [batch*seq, hidden]
        qkv_act = self.qkv(x)
        query, key, value = ops.split(qkv_act, 3, axis=-1)

        #: [batch, seq, heads, head_size]
        query = reshape_for_scores(query, self.config.model.sequence_length, self.n_heads)
        key = reshape_for_scores(key, self.config.model.sequence_length, self.n_heads)
        value = reshape_for_scores(value, self.config.model.sequence_length, self.n_heads)

        sin, cos = trig_table_constants(
            self.config.model.sequence_length,
            self.rotary_dim,
            self.config.model.attention.rotary_positional_embeddings_base,
            self.config.model.dtype,
        )

        query = rotary_pos_embed(query, sin, cos, self.rotary_dim).transpose((0, 2, 1, 3))
        key = rotary_pos_embed(key, sin, cos, self.rotary_dim).transpose((0, 2, 3, 1))
        value = value.transpose((0, 2, 1, 3))

        causal_mask = popxl.constant(
            # HF version 1e9 to mask. However, this model runs in float16 and 1e9 is beyond the float16 range, therefore 1e4 is used to similar effect.
            1e4 * (np.tril(np.ones((self.config.model.sequence_length, self.config.model.sequence_length))) - 1),
            query.dtype,
            name="causal_mask",
        )

        attn_output = self.attention_block(query, key, value, causal_mask)

        return attn_output.transpose((0, 2, 1, 3)).reshape(
            (self.config.execution.micro_batch_size * self.config.model.sequence_length, -1)
        )

    def attention_block(self, query: popxl.Tensor, key: popxl.Tensor, value: popxl.Tensor, mask: popxl.Tensor):
        attn_weights = query @ key

        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))
        attn_weights = attn_weights + mask

        attn_scores = ops.softmax(attn_weights, axis=-1)

        return attn_scores @ value


class DollySelfAttentionTP(addons.Module):
    def __init__(self, config: DollyConfig):
        super().__init__()

        self.config = config
        attn_tp = (
            config.execution.tensor_parallel
            if config.execution.attention_tensor_parallel is None
            else config.execution.attention_tensor_parallel
        )
        tp = attn_tp
        dp = config.execution.data_parallel * (config.execution.tensor_parallel // attn_tp)
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

        # Sharded across devices
        self.heads = DollyAttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices
        self.output = Linear(self.config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""

        # ----- Sharded computation -----
        z = self.heads(x)
        z = self.output(z)

        z = replicated_all_reduce(z, group=self.replica_grouping.transpose())

        self.output_bias = self.add_variable_input(
            "output_bias",
            lambda: np.zeros(z.shape[-1]),
            z.dtype,
        )
        z = z + self.output_bias

        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        hidden_dim = config.model.hidden_size
        attn_tp = (
            config.execution.tensor_parallel
            if config.execution.attention_tensor_parallel is None
            else config.execution.attention_tensor_parallel
        )
        heads = config.model.attention.heads

        qkv_w = hf_model.query_key_value.weight.data.T.reshape(hidden_dim, heads, 3, -1)
        hf_query_w, hf_key_w, hf_value_w = np.split(qkv_w, 3, axis=-2)
        hf_query_w, hf_key_w, hf_value_w = map(
            lambda p: to_numpy(p, dtype).reshape(hidden_dim, hidden_dim), (hf_query_w, hf_key_w, hf_value_w)
        )
        query_w, key_w, value_w = map(lambda p: shard(p, attn_tp, axis=-1), (hf_query_w, hf_key_w, hf_value_w))
        qkv_weight = np.concatenate((query_w, key_w, value_w), axis=-1)
        # qkv_weight = repeat_axis(qkv_weight, n=repeat_tp, axis=0)

        qkv_b = hf_model.query_key_value.bias.data.reshape(heads, 3, -1)
        hf_query_b, hf_key_b, hf_value_b = np.split(qkv_b, 3, axis=1)

        hf_query_b, hf_key_b, hf_value_b = map(
            lambda p: to_numpy(p, dtype).reshape(-1), (hf_query_b, hf_key_b, hf_value_b)
        )
        query_b, key_b, value_b = map(lambda p: np.split(p, attn_tp, axis=0), (hf_query_b, hf_key_b, hf_value_b))
        qkv_bias = np.concatenate((query_b, key_b, value_b), axis=1)
        # qkv_bias = repeat_axis(qkv_bias, n=repeat_tp, axis=0)

        out_proj_w = to_numpy(hf_model.dense.weight.data.T, dtype)

        weights = {
            variables.heads.qkv.weight: qkv_weight.squeeze(),
            variables.heads.qkv.bias: qkv_bias.squeeze(),
            variables.output.weight: shard(out_proj_w, attn_tp, axis=0).squeeze(),
            variables.output_bias: to_numpy(hf_model.dense.bias.data, dtype),
        }
        return weights
