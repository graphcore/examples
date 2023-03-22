# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import math
from typing import Optional, Dict

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention

import popxl_addons as addons
from config import BertConfig
from popxl_addons import NamedTensors
from popxl_addons.layers import Linear, LayerNorm
import numpy as np


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    assert len(x.shape) == 2
    micro_batch_size = x.shape[0] // sequence_length
    head_hidden_size = x.shape[1] // heads
    return x.reshape((micro_batch_size, sequence_length, heads, head_hidden_size))


def transpose_for_scores(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
    assert len(x.shape) == 4
    perm = (0, 2, 1, 3) if not is_key else (0, 2, 3, 1)
    return x.transpose(perm)


class AttentionHeads(addons.Module):
    def __init__(self, config: BertConfig, replica_grouping: Optional[ReplicaGrouping] = None):
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

    def build(self, x: popxl.Tensor, mask: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        mask = mask.reshape((self.config.execution.micro_batch_size, 1, 1, self.config.model.sequence_length))

        qkv_act = self.qkv(x)
        q_act, k_act, v_act = ops.split(qkv_act, 3, axis=-1)

        def transform_heads(x: popxl.Tensor, is_key: bool) -> popxl.Tensor:
            return transpose_for_scores(reshape_for_scores(x, self.config.model.sequence_length, self.n_heads), is_key)

        q_act = transform_heads(q_act, False)
        k_act = transform_heads(k_act, True)
        v_act = transform_heads(v_act, False)

        v = q_act @ k_act

        v = v * (1 / math.sqrt(q_act.shape[-1]))
        v = v + ((mask - 1) * 1000.0)
        v_scores = ops.softmax(v, axis=-1)

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            v_scores = ops.dropout(v_scores, seed, p=self.config.model.dropout_prob)

        z = v_scores @ v_act

        x_part_shape = list(x.shape)
        x_part_shape[-1] = x_part_shape[-1] // self.n_heads_groups
        z = z.transpose((0, 2, 1, 3)).reshape(x_part_shape)

        return z

    @staticmethod
    def hf_mapping(
        config: BertConfig, vars: NamedTensors, hf_model: BertSelfAttention
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        query_w = to_numpy(hf_model.query.weight, dtype).T
        key_w = to_numpy(hf_model.key.weight, dtype).T
        value_w = to_numpy(hf_model.value.weight, dtype).T
        query_b = to_numpy(hf_model.query.bias, dtype)
        key_b = to_numpy(hf_model.key.bias, dtype)
        value_b = to_numpy(hf_model.value.bias, dtype)

        return {
            vars.qkv.weight: np.ascontiguousarray(np.concatenate((query_w, key_w, value_w), axis=-1)),
            vars.qkv.bias: np.concatenate((query_b, key_b, value_b), axis=-1),
        }


class SelfAttention(addons.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.heads = AttentionHeads(config)
        self.output = Linear(self.config.model.hidden_size)
        self.norm = LayerNorm()

    def build(self, x: popxl.Tensor, mask: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        heads_seed = None
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            seed, heads_seed = ops.split_random_seed(seed)

        z = self.heads(x, mask, heads_seed)
        z = self.output(z)

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        z = self.norm(z + x)
        return z

    @staticmethod
    def hf_mapping(
        config: BertConfig, variables: NamedTensors, hf_model: BertAttention
    ) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        return {
            variables.output.weight: np.ascontiguousarray(to_numpy(hf_model.output.dense.weight, dtype).T),
            variables.output.bias: to_numpy(hf_model.output.dense.bias, dtype),
            variables.norm.weight: to_numpy(hf_model.output.LayerNorm.weight, dtype),
            variables.norm.bias: to_numpy(hf_model.output.LayerNorm.bias, dtype),
            **AttentionHeads.hf_mapping(config, variables.heads, hf_model.self),
        }
