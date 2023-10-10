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
from popxl_addons.layers import Linear

from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons.array_munging import shard

from modelling.embedding import T5RelPosEmbeddingsTP

from config import T5Config
from transformers.models.t5.modeling_t5 import T5Attention as HFModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF


def reshape_for_scores(x: popxl.Tensor, sequence_length: int, heads: int) -> popxl.Tensor:
    assert len(x.shape) == 2
    micro_batch_size = x.shape[0] // sequence_length
    head_hidden_size = x.shape[1] // heads
    return x.reshape((micro_batch_size, sequence_length, heads, head_hidden_size))


class T5AttentionHeads(addons.Module):
    def __init__(self, config: T5Config, replica_grouping: Optional[ReplicaGrouping] = None):
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
        self.inner_dim = self.config.model.attention.heads * self.config.model.attention.d_kv
        self.qkv = Linear(3 * self.inner_dim // n_heads_groups, replica_grouping=replica_grouping, bias=False)
        self.relative_attention_num_buckets = self.config.model.attention.relative_attention_num_buckets
        self.max_distance = self.config.model.attention.relative_attention_max_distance
        self.rel_pos_embedding = T5RelPosEmbeddingsTP(
            self.config.model.dtype,
            self.relative_attention_num_buckets,
            self.n_heads,
            replica_grouping=self.replica_grouping,
        )
        self.seq_len = self.config.model.sequence_length

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        is_encoder: popxl.Tensor,
        rel_pos_weight: Optional[popxl.Tensor] = None,
        seed: Optional[popxl.Tensor] = None,
    ):
        # x: [batch*seq, hidden]
        # mask: [batch*seq]
        qkv_act = self.qkv(x)
        query, key, value = ops.split(qkv_act, 3, axis=-1)
        # inner' = inner // n_heads_groups
        # query: [batch*seq, inner'] (key and value have the same shape)

        query = reshape_for_scores(query, self.seq_len, self.n_heads)
        key = reshape_for_scores(key, self.seq_len, self.n_heads)
        value = reshape_for_scores(value, self.seq_len, self.n_heads)
        # head_size = inner' // heads = d_kv
        # query, key, value: [batch, seq, heads, head_size]
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 3, 1))
        value = value.transpose((0, 2, 1, 3))
        # query, value: [batch, heads, seq, head_size]
        # key: [batch, heads, head_size, seq]

        input_ids = self.compute_relative_position_indices(is_encoder)
        rel_pos_encodings = self.rel_pos_embedding(input_ids, rel_pos_weight)

        # rel_pos_encodings: [seq, seq, heads]
        rel_pos_encodings = rel_pos_encodings.transpose((2, 0, 1)).reshape(
            (1, self.n_heads, self.seq_len, self.seq_len)
        )
        # rel_pos_encodings: [1, heads, seq, seq]

        mask = mask.reshape((self.config.execution.micro_batch_size, 1, 1, self.seq_len))
        # mask: [batch, 1, 1, seq]
        causal_mask = np.tril(np.ones((self.seq_len, self.seq_len)))
        # Get the complementary mask (with 1s and 0s switched wrt the causal mask)
        compl_mask = 1 - causal_mask
        causal_mask = popxl.constant(causal_mask, query.dtype, name="causal_mask")
        compl_mask = popxl.constant(compl_mask, query.dtype, name="compl_mask")
        # For the encoder is_encoder == 1, and for the decoder is_encoder == 0
        # We want the encoder to have a causal mask of all ones,
        # and the decoder to have a triangular causal mask
        causal_mask = causal_mask + is_encoder * compl_mask
        causal_mask = causal_mask.reshape((1, 1, self.seq_len, self.seq_len))
        # causal_mask: [1, 1, seq, seq]
        # Combine the two masks
        mask = mask * causal_mask
        # turn the values 1 to 0, and values 0 to a large negative number,
        # making sure the number can be represented in the current dtype
        large_num = 1e9 if mask.dtype == popxl.float32 else 1e4
        mask = (mask - 1) * large_num

        if self.config.execution.attention_serialisation > 1:
            queries = ops.split(query, self.config.execution.attention_serialisation, axis=2)
            rel_pos_encs = ops.split(rel_pos_encodings, self.config.execution.attention_serialisation, axis=2)
            masks = ops.split(mask, self.config.execution.attention_serialisation, axis=2)

            blk_graph = popxl.gcg().ir.create_graph(
                self.attention_block, queries[0], key, value, rel_pos_encs[0], masks[0], seed
            )

            attn_outputs = []
            for query_i, rel_pos_enc_i, mask_i in zip(queries, rel_pos_encs, masks):
                args = [query_i, key, value, rel_pos_enc_i, mask_i]
                # Each step should have different dropout
                if seed is not None:
                    seed, blk_seed = ops.split_random_seed(seed)
                    args.append(blk_seed)

                (attn_block_output,) = ops.call(blk_graph, *args)
                # attn_block_output: [batch, heads, seq', head_size]

                attn_outputs.append(attn_block_output)
            attn_output = ops.concat(attn_outputs, axis=2)
            # attn_output: [batch, heads, seq, head_size]
        else:
            attn_output = self.attention_block(query, key, value, rel_pos_encodings, mask, seed)

        return attn_output.transpose((0, 2, 1, 3)).reshape((self.config.execution.micro_batch_size * self.seq_len, -1))

    def attention_block(
        self,
        query: popxl.Tensor,
        key: popxl.Tensor,
        value: popxl.Tensor,
        rel_pos_enc: popxl.Tensor,
        mask: popxl.Tensor,
        seed: popxl.Tensor,
    ):
        # seq' = seq // attention_serialisation
        # query: [batch, heads, seq', head_size]
        # key: [batch, heads, head_size, seq]
        attn_weights = query @ key
        # attn_weights: [batch, heads, seq', seq]
        # rel_pos_enc: [1, heads, seq', seq]
        # mask: [batch, 1, seq', seq]
        attn_weights = attn_weights + rel_pos_enc + mask

        attn_scores = ops.softmax(attn_weights, axis=-1)
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            attn_scores = ops.dropout(attn_scores, seed, p=self.config.model.dropout_prob)
        # value: [batch, heads, seq, head_size]
        return attn_scores @ value

    def compute_relative_position_indices(self, is_encoder: popxl.Tensor):
        # We need to perform the same computation for encoder and decoder,
        # we use the scalar tensor is_encoder to extract the slice of
        # relative positions for the current layer.
        # Get the indices relative to the encoder
        rel_pos_buckets_enc = self._compute_relative_position_buckets(True)
        # Get the indices relative to the decoder
        rel_pos_buckets_dec = self._compute_relative_position_buckets(False)
        input_ids_enc = popxl.constant(rel_pos_buckets_enc, popxl.int32, "rel_pos_buckets_enc")
        input_ids_dec = popxl.constant(rel_pos_buckets_dec, popxl.int32, "rel_pos_buckets_dec")
        # Combine them to produce the correct indices
        is_encoder = ops.cast(is_encoder, popxl.int32)
        input_ids = is_encoder * input_ids_enc + (1 - is_encoder) * input_ids_dec
        return input_ids

    def _compute_relative_position_buckets(self, is_encoder: bool):
        # This numpy computation is largely inspired by the equivalent torch implementation from HuggingFace
        query_position = np.arange(self.seq_len, dtype=np.int32)[:, np.newaxis]
        key_position = np.arange(self.seq_len, dtype=np.int32)[np.newaxis, :]
        # matrix of all the relative distances
        relative_position = key_position - query_position
        if is_encoder:
            # half of the buckets for positions behind, half for positions ahead
            num_buckets = self.relative_attention_num_buckets // 2
            # offset the positions ahead by num_buckets
            relative_buckets = (relative_position > 0).astype(np.int32) * num_buckets
            relative_position = np.abs(relative_position)
        else:
            # all of the buckets for positions behind
            num_buckets = self.relative_attention_num_buckets
            relative_buckets = np.zeros_like(relative_position)
            relative_position = -np.minimum(relative_position, relative_buckets)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        # Ignore divide by zero warning: it is expected and those entries
        # will be overwritten in the np.where below
        with np.errstate(divide="ignore"):
            relative_position_if_large = max_exact + (
                np.log(relative_position.astype(np.float32) / max_exact)
                / math.log(self.max_distance / max_exact)
                * (num_buckets - max_exact)
            ).astype(np.int32)
        relative_position_if_large = np.minimum(
            relative_position_if_large, np.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += np.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


class T5SelfAttentionTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()

        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

        # Sharded across devices
        self.heads = T5AttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices
        self.output = Linear(self.config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        is_encoder: popxl.Tensor,
        rel_pos_weight: Optional[popxl.Tensor] = None,
        seed: Optional[popxl.Tensor] = None,
    ) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""
        heads_seed = None
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            seed, heads_seed = ops.split_random_seed(seed)

        # ----- Identical computation -----
        z = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----
        z = self.heads(z, mask, is_encoder, rel_pos_weight, seed=heads_seed)
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        hf_query_w = to_numpy(hf_model.q.weight.data, dtype).T
        hf_key_w = to_numpy(hf_model.k.weight.data, dtype).T
        hf_value_w = to_numpy(hf_model.v.weight.data, dtype).T

        query_w = shard(hf_query_w, n_shards, -1)
        key_w = shard(hf_key_w, n_shards, -1)
        value_w = shard(hf_value_w, n_shards, axis=-1)

        # Only the first layer of the encoder or decoder stack
        # will have the original weights for the rel pos embedding
        do_rel_pos = hf_model.has_relative_attention_bias and "rel_pos_embedding" in variables.heads
        if do_rel_pos:
            rel_pos_w = to_numpy(hf_model.relative_attention_bias.weight.data, dtype).T

        out_proj_w = to_numpy(hf_model.o.weight.data.T, dtype)

        weights = {
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
        if do_rel_pos:
            weights[variables.heads.rel_pos_embedding.weight] = shard(rel_pos_w, n_shards, axis=0).transpose((0, 2, 1))
        return weights

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        q, k, v = np.split(variables_data.heads.qkv.weight, 3, axis=-1)
        state_dict = {}
        state_dict["q.weight"] = torch.tensor(np.concatenate(q.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype)
        state_dict["k.weight"] = torch.tensor(np.concatenate(k.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype)
        state_dict["v.weight"] = torch.tensor(np.concatenate(v.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype)
        # Only the first layer of the encoder or decoder stack
        # will have the original weights for the rel pos embedding
        if "rel_pos_embedding" in variables_data.heads:
            state_dict["relative_attention_bias.weight"] = torch.tensor(
                np.concatenate(variables_data.heads.rel_pos_embedding.weight.transpose((0, 2, 1)), axis=0).T,
                dtype=config.torch_dtype,
            )
        state_dict["o.weight"] = torch.tensor(
            np.concatenate(variables_data.output.weight, axis=0).T, dtype=config.torch_dtype
        )
        return state_dict


class T5CrossAttentionHeads(addons.Module):
    def __init__(self, config: T5Config, replica_grouping: Optional[ReplicaGrouping] = None):
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
        self.inner_dim = self.config.model.attention.heads * self.config.model.attention.d_kv
        self.q = Linear(self.inner_dim // n_heads_groups, replica_grouping=replica_grouping, bias=False)
        self.kv = Linear(2 * self.inner_dim // n_heads_groups, replica_grouping=replica_grouping, bias=False)
        self.seq_len = self.config.model.sequence_length

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        enc_output: popxl.Tensor,
        seed: Optional[popxl.Tensor] = None,
    ):
        # x: [batch*seq, hidden]
        # mask: [batch*seq]
        # Get the query from the x
        query = self.q(x)
        # And get the key and value from the encoder output
        kv_act = self.kv(enc_output)
        key, value = ops.split(kv_act, 2, axis=-1)
        # inner' = inner // n_heads_groups
        # query: [batch*seq, inner'] (key and value have the same shape)

        query = reshape_for_scores(query, self.seq_len, self.n_heads)
        key = reshape_for_scores(key, self.seq_len, self.n_heads)
        value = reshape_for_scores(value, self.seq_len, self.n_heads)
        # head_size = inner' // heads = d_kv
        # query, key, value: [batch, seq, heads, head_size]
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 3, 1))
        value = value.transpose((0, 2, 1, 3))
        # query, value: [batch, heads, seq, head_size]
        # key: [batch, heads, head_size, seq]

        # There is no relative position encoding in the cross attention

        mask = mask.reshape((self.config.execution.micro_batch_size, 1, 1, self.seq_len))
        # mask: [batch, 1, 1, seq]
        # turn the values 1 to 0, and values 0 to a large negative number,
        # making sure the number can be represented in the current dtype
        large_num = 1e9 if mask.dtype == popxl.float32 else 1e4
        mask = (mask - 1) * large_num

        if self.config.execution.attention_serialisation > 1:
            queries = ops.split(query, self.config.execution.attention_serialisation, axis=2)

            blk_graph = popxl.gcg().ir.create_graph(self.attention_block, queries[0], key, value, mask, seed)

            attn_outputs = []
            for query_i in queries:
                args = [query_i, key, value, mask]
                # Each step should have different dropout
                if seed is not None:
                    seed, blk_seed = ops.split_random_seed(seed)
                    args.append(blk_seed)

                (attn_block_output,) = ops.call(blk_graph, *args)
                # attn_block_output: [batch, heads, seq', head_size]

                attn_outputs.append(attn_block_output)
            attn_output = ops.concat(attn_outputs, axis=2)
            # attn_output: [batch, heads, seq, head_size]
        else:
            attn_output = self.attention_block(query, key, value, mask, seed)

        return attn_output.transpose((0, 2, 1, 3)).reshape((self.config.execution.micro_batch_size * self.seq_len, -1))

    def attention_block(
        self,
        query: popxl.Tensor,
        key: popxl.Tensor,
        value: popxl.Tensor,
        mask: popxl.Tensor,
        seed: popxl.Tensor,
    ):
        # seq' = seq // attention_serialisation
        # query: [batch, heads, seq', head_size]
        # key: [batch, heads, head_size, seq]
        attn_weights = query @ key
        # attn_weights: [batch, heads, seq', seq]
        # mask: [batch, 1, 1, seq]
        attn_weights = attn_weights + mask

        attn_scores = ops.softmax(attn_weights, axis=-1)
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            attn_scores = ops.dropout(attn_scores, seed, p=self.config.model.dropout_prob)
        # value: [batch, heads, seq, head_size]
        return attn_scores @ value


class T5CrossAttentionTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()

        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

        # Sharded across devices
        self.heads = T5CrossAttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices
        self.output = Linear(self.config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(
        self,
        x: popxl.Tensor,
        mask: popxl.Tensor,
        enc_output: popxl.Tensor,
        seed: Optional[popxl.Tensor] = None,
    ) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""
        heads_seed = None
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            seed, heads_seed = ops.split_random_seed(seed)

        # ----- Identical computation -----
        z = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())
        enc_z = replicated_all_reduce_identical_inputs(enc_output, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----
        z = self.heads(z, mask, enc_z, seed=heads_seed)
        z = self.output(z)

        z = replicated_all_reduce_identical_grad_inputs(z, group=self.replica_grouping.transpose())

        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            z = ops.dropout(z, seed, p=self.config.model.dropout_prob)

        return z

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel

        hf_query_w = to_numpy(hf_model.q.weight.data, dtype).T
        hf_key_w = to_numpy(hf_model.k.weight.data, dtype).T
        hf_value_w = to_numpy(hf_model.v.weight.data, dtype).T

        query_w = shard(hf_query_w, n_shards, -1)
        key_w = shard(hf_key_w, n_shards, -1)
        value_w = shard(hf_value_w, n_shards, axis=-1)

        out_proj_w = to_numpy(hf_model.o.weight.data.T, dtype)

        return {
            variables.heads.q.weight: query_w,
            variables.heads.kv.weight: np.ascontiguousarray(
                np.concatenate(
                    [np.concatenate([key_w[i], value_w[i]], axis=-1)[np.newaxis, ...] for i in range(n_shards)]
                )
            ),
            variables.output.weight: shard(out_proj_w, n_shards, axis=0),
        }

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        q = variables_data.heads.q.weight
        k, v = np.split(variables_data.heads.kv.weight, 2, axis=-1)
        state_dict = {}
        state_dict["q.weight"] = torch.tensor(np.concatenate(q.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype)
        state_dict["k.weight"] = torch.tensor(np.concatenate(k.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype)
        state_dict["v.weight"] = torch.tensor(np.concatenate(v.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype)
        state_dict["o.weight"] = torch.tensor(
            np.concatenate(variables_data.output.weight, axis=0).T, dtype=config.torch_dtype
        )

        return state_dict
