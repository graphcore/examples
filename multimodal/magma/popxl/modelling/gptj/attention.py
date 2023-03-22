# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
import math
import torch

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.named_tensors import NamedTensorData
from popxl_addons.array_munging import shard
from popxl_addons.layers import Linear

from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons.ops.rotary_pos_embed import rotary_pos_embed, trig_table_constants

from configs import GPTJConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoAttention as HFModel
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig as GPTJConfigHF
from functools import partial


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    assert len(x.shape) == 2
    micro_batch_size = x.shape[0] // sequence_length
    head_hidden_size = x.shape[1] // heads
    return x.reshape((micro_batch_size, sequence_length, heads, head_hidden_size))


class GPTJAttentionHeads(addons.Module):
    def __init__(self, config: GPTJConfig, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.config = config
        self.replica_grouping = replica_grouping

        if self.replica_grouping:
            n_heads_groups = self.replica_grouping.num_groups
        else:
            n_heads_groups = 1

        assert (
            self.config.attention.heads % n_heads_groups == 0
        ), f"{self.config.attention.heads} % {n_heads_groups} != 0"

        self.n_heads_groups = n_heads_groups
        self.n_heads = self.config.attention.heads // n_heads_groups
        self.qkv = Linear(3 * self.config.hidden_size // n_heads_groups, replica_grouping=replica_grouping, bias=False)
        self.rotary_dim = self.config.attention.rotary_dim or self.config.hidden_size // self.n_heads
        self.head_size = self.config.hidden_size // self.n_heads

    def build(self, x: popxl.Tensor):
        # optimised inference
        bs = self.config.execution.micro_batch_size

        # x: [batch*seq, hidden]
        qkv_act: popxl.Tensor = self.qkv(x)  # type: ignore
        query, key, value = ops.split(qkv_act, 3, axis=-1)

        query, key = self.apply_rotary(query, key)

        #: [batch, seq, heads, head_size]
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 3, 1))
        value = reshape_for_scores(value, self.config.sequence_length, self.n_heads).transpose((0, 2, 1, 3))

        causal_mask = popxl.constant(
            # HF uses 1e9 which is beyond fp16 range
            1e4 * (np.tril(np.ones((self.config.sequence_length, self.config.sequence_length))) - 1),
            query.dtype,
            name="causal_mask",
        )

        if self.config.execution.attention_serialisation > 1:
            queries = ops.split(query, self.config.execution.attention_serialisation, axis=2)
            masks = ops.split(causal_mask, self.config.execution.attention_serialisation, axis=0)

            blk_graph = popxl.gcg().ir.create_graph(self.attention_block, queries[0], key, value, masks[0])

            attn_outputs = []
            for query_i, mask_i in zip(queries, masks):
                args = [query_i, key, value, mask_i]
                (attn_block_output,) = ops.call(blk_graph, *args)

                attn_outputs.append(attn_block_output)
            attn_output = ops.concat(attn_outputs, axis=2)
        else:
            attn_output = self.attention_block(query, key, value, causal_mask)

        return attn_output.transpose((0, 2, 1, 3)).reshape((bs * self.config.sequence_length, -1))

    def apply_rotary(self, query: popxl.Tensor, key: popxl.Tensor):
        sin, cos = trig_table_constants(
            self.config.sequence_length,
            self.rotary_dim,
            self.config.attention.rotary_positional_embeddings_base,
            self.config.dtype,
        )
        query = reshape_for_scores(query, self.config.sequence_length, self.n_heads)
        key = reshape_for_scores(key, self.config.sequence_length, self.n_heads)
        query = rotary_pos_embed(query, sin, cos, self.rotary_dim)
        key = rotary_pos_embed(key, sin, cos, self.rotary_dim)

        return query, key

    def attention_block(self, query: popxl.Tensor, key: popxl.Tensor, value: popxl.Tensor, mask: popxl.Tensor):
        attn_weights = query @ key
        attn_weights = attn_weights * (1.0 / math.sqrt(value.shape[-1]))
        attn_weights = attn_weights + mask
        attn_scores = ops.softmax(attn_weights, axis=-1)
        return attn_scores @ value

    @staticmethod
    def finetuneanon_mapping(
        config: GPTJConfig, vars: NamedTensors, hf_model: HFModel
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype

        query_w = to_numpy(hf_model.q_proj.weight.data, dtype).T
        key_w = to_numpy(hf_model.k_proj.weight.data, dtype).T
        value_w = to_numpy(hf_model.v_proj.weight.data, dtype).T

        return {vars.qkv.weight: np.ascontiguousarray(np.concatenate((query_w, key_w, value_w), axis=-1))}


class GPTJSelfAttentionTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()

        self.config = config
        tp = config.execution.tensor_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=1)

        # Sharded across devices
        self.heads = GPTJAttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices
        self.output = Linear(self.config.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""
        # ----- Identical computation -----
        z = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----
        z = self.heads(z)
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())
        return z

    @staticmethod
    def finetuneanon_mapping(config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.dtype
        n_shards = config.execution.tensor_parallel

        hf_query_w = to_numpy(hf_model.attention.q_proj.weight.data, dtype).T
        hf_key_w = to_numpy(hf_model.attention.k_proj.weight.data, dtype).T
        hf_value_w = to_numpy(hf_model.attention.v_proj.weight.data, dtype).T

        query_w = shard(hf_query_w, n_shards, -1)
        key_w = shard(hf_key_w, n_shards, -1)
        value_w = shard(hf_value_w, n_shards, axis=-1)

        out_proj_w = to_numpy(hf_model.attention.out_proj.weight.data.T, dtype)

        return {
            variables.heads.qkv.weight: np.ascontiguousarray(
                np.concatenate(
                    [
                        np.concatenate([query_w[i], key_w[i], value_w[i]], axis=-1)[np.newaxis, ...]
                        for i in range(n_shards)
                    ]
                )
            ),
            variables.output.weight: shard(out_proj_w, n_shards, axis=0),
        }
