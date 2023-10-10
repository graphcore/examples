# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict, Tuple
import math

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.array_munging import shard, pad_axis
from popxl_addons.layers import Linear
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce

from .rotary_pos_embed import rotary_pos_embed, trig_table_constants

from config import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention as HFModel


class LlamaAttentionHeads(addons.Module):
    def __init__(self, config: LlamaConfig, replica_grouping: Optional[ReplicaGrouping] = None):
        super().__init__()
        self.config = config
        self.replica_grouping = replica_grouping

        if self.replica_grouping:
            n_heads_groups = self.replica_grouping.num_groups
        else:
            n_heads_groups = 1

        self.n_heads_groups = n_heads_groups

        n_unsharded_heads = self.config.model.attention.heads
        kv_unsharded_heads = self.config.model.attention.kv_heads

        if n_unsharded_heads % n_heads_groups != 0:
            self.padding = True
            if n_unsharded_heads > n_heads_groups:
                n_unsharded_heads += n_unsharded_heads % n_heads_groups

        # TODO: Need to pad KV heads for GQA to use higher TP factor (unsharded heads < groups)

        assert n_unsharded_heads % n_heads_groups == 0, f"N heads: {n_unsharded_heads} % {n_heads_groups} != 0"
        assert kv_unsharded_heads % n_heads_groups == 0, f"KV heads: {kv_unsharded_heads} % {n_heads_groups} != 0"

        # This is the number of KV pairs per query for grouped-query attention
        self.gq_groups = None
        if kv_unsharded_heads != n_unsharded_heads:
            self.gq_groups = self.config.model.attention.heads // self.config.model.attention.kv_heads

        # Number of heads and KV heads per replica
        self.n_heads = n_unsharded_heads // n_heads_groups
        self.kv_heads = kv_unsharded_heads // n_heads_groups

        # If N heads and hidden dim (Hd), each heads actual dim= N/Hd without padding (assume no sharding)
        # To enable sharding for N/M heads in case where N % groups != 0:
        # padded_heads/groups (i.e. heads per replica) multiplied N/Hd rather than padded_heads/Hd
        self.dim_head = self.config.model.hidden_size // self.config.model.attention.heads

        # Llama Attention does not use bias - separate Q and KV to allow for different K,V weights for GQA
        self.q = Linear(self.dim_head * self.n_heads, bias=False, replica_grouping=replica_grouping)
        self.kv = Linear(2 * (self.dim_head * self.kv_heads), bias=False, replica_grouping=replica_grouping)

        # Rotary dims determined by hidden size and attention head as in Transformers Llama implementation.
        # No rotary scaling percentage is implemented for the model.
        self.rotary_ndims = self.dim_head

        # Rotary positional embeddings base value used as constant in Transformers Llama.
        self.rotary_pos_emb_base = 10000

        # For processing 'batches' as token-by-token sequences
        self.num_seqs = self.config.execution.micro_batch_size

        # Set an 'initial' sequence length (1 token per run for KV cache) for reshapes consistency
        self.initial_seq_len = self.config.model.sequence_length
        if self.config.execution.use_cache:
            self.initial_seq_len = 1

    def repeat_kv(self, k: popxl.Tensor, v: popxl.Tensor) -> Tuple[popxl.Tensor, popxl.Tensor]:
        """
        Similar to torch.expand - multiply key by ones tensor with expanded shape. Broadcasting only requires affected
        keys to be specified in ones tensor

        - K/V is expanded to (bs, groups, kv_heads, *, *) creating a generic 'repeat' order [a,b] => [[a,b],[a,b]]
        - Transpose to rearrange to (bs, kv_heads, groups, *, *) creating an 'interleaved' order [[a,b],[a,b]] => [[a,a],[b,b]]
        - Then reshape equivalent to a 'squeeze' on dim 1 => [[a,a,b,b]] in all creating an interleaved repeat.
        """

        expand_mask = popxl.constant(
            np.ones((1, self.gq_groups, self.kv_heads, 1, 1)), self.config.model.dtype, "expand_mask"
        )

        grouped_key = (k * expand_mask).transpose((0, 2, 1, 3, 4)).reshape_((self.num_seqs, self.n_heads, *k.shape[2:]))
        grouped_value = (
            (v * expand_mask).transpose((0, 2, 1, 3, 4)).reshape_((self.num_seqs, self.n_heads, *v.shape[2:]))
        )

        return grouped_key, grouped_value

    def per_token_causal_mask(self, last_token_indices: popxl.Tensor) -> popxl.Tensor:
        """
        Generate a causal mask per token: creates an `np.arange` tensor of range = sequence length
        """
        range_mask = popxl.constant(
            np.repeat(np.arange(self.config.model.sequence_length)[None, ...], self.num_seqs, axis=0),
            self.config.model.dtype,
            "range_mask",
        )

        mask_low = popxl.constant(-1.0e4, self.config.model.dtype, "mask_low")
        mask_zero = popxl.constant(0.0, self.config.model.dtype, "mask_zero")
        causal_mask = ops.where(ops.greater(range_mask, last_token_indices), mask_low, mask_zero)

        # Requires reshape for extra dim when BS > 1
        if self.config.execution.micro_batch_size > 1:
            causal_mask = causal_mask.reshape((causal_mask.shape[0], 1, 1, causal_mask.shape[1]))

        return causal_mask

    def indv_trig_table_constants(self, last_token_indices: popxl.Tensor) -> Tuple[popxl.Tensor, popxl.Tensor]:
        """
        Generate cosΦ, sinΦ np data of shape (1, rotary_dim) for current token index `tok_idx`, where
        Φ = m * θ_i,
        θ_i = base^(-2i/rotary_dim), i = 0 ... rotary_dim
        """
        theta = popxl.constant(
            self.rotary_pos_emb_base ** (-1 * np.arange(0, self.rotary_ndims, 2) / self.rotary_ndims),
            self.config.model.dtype,
            "theta",
        )

        phi = last_token_indices * theta
        cos = ops.cos(phi)
        sin = ops.sin(phi)

        # Requires reshape for extra dim when BS > 1
        if self.config.execution.micro_batch_size > 1:
            cos = cos.reshape((self.config.execution.micro_batch_size, 1, -1))
            sin = sin.reshape((self.config.execution.micro_batch_size, 1, -1))

        return sin, cos

    def attention_block(self, query: popxl.Tensor, key: popxl.Tensor, value: popxl.Tensor, mask: popxl.Tensor):
        attn_weights = query @ key

        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))

        attn_weights = attn_weights + mask

        if attn_weights.dtype == popxl.float16:
            attn_weights = ops.cast(attn_weights, popxl.float32)

        attn_scores = ops.softmax(attn_weights, axis=-1)

        if attn_scores.dtype == popxl.float32 and self.config.model.dtype == popxl.float16:
            attn_scores = ops.cast(attn_scores, popxl.float16)

        return attn_scores @ value

    def build(
        self,
        x: popxl.Tensor,
        last_token_indices: Optional[popxl.Tensor] = None,
        past_k: Optional[popxl.Tensor] = None,
        past_v: Optional[popxl.Tensor] = None,
    ):

        # x: [batch*seq, hidden]
        kv = self.kv(x)

        key, value = ops.split(kv, 2, axis=-1)
        query = self.q(x)

        query = query.reshape((self.num_seqs, self.initial_seq_len, self.n_heads, query.shape[1] // self.n_heads))
        key = key.reshape((self.num_seqs, self.initial_seq_len, self.kv_heads, key.shape[1] // self.kv_heads))
        value = value.reshape((self.num_seqs, self.initial_seq_len, self.kv_heads, value.shape[1] // self.kv_heads))

        if not self.config.execution.use_cache:
            causal_mask = popxl.constant(
                # HF version 1e9 to mask. However, this model runs in float16 and 1e9 is beyond the
                # float16 range, therefore 1e4 is used to similar effect.
                1e4 * (np.tril(np.ones((self.config.model.sequence_length, self.config.model.sequence_length))) - 1),
                query.dtype,
                name="causal_mask",
            )

            sin, cos = trig_table_constants(
                self.config.model.sequence_length,
                self.rotary_ndims,
                self.rotary_pos_emb_base,
                self.config.model.dtype,
            )
        else:
            # Cast last token indices to model dtype for causal mask generation and trig table constants
            # For batch size > 1, reshape is equivalent to unsqueeze/expand_dims (Since unused '1' dims
            # are squeezed when building graph)
            lti = ops.cast(last_token_indices, self.config.model.dtype)
            lti = lti.reshape((last_token_indices.shape[0], 1))
            # Generate a [bs, slen] size per-token causal mask in-graph
            causal_mask = self.per_token_causal_mask(lti)

            # We can get the sin and cos tensors for a single token directly in-graph
            sin, cos = self.indv_trig_table_constants(lti)

        query = rotary_pos_embed(query, sin, cos, self.rotary_ndims).transpose((0, 2, 1, 3))
        key = rotary_pos_embed(key, sin, cos, self.rotary_ndims).transpose((0, 2, 3, 1))
        value = value.transpose((0, 2, 1, 3))

        # Update the buffer values
        if self.config.execution.use_cache:
            if self.config.execution.micro_batch_size > 1:
                lti = ops.cast(lti, popxl.int32)

                curr_k = ops.dynamic_update_(
                    past_k, lti, key, [0, 3], [self.config.execution.micro_batch_size, 1], True
                )
                curr_v = ops.dynamic_update_(
                    past_v, lti, value, [0, 2], [self.config.execution.micro_batch_size, 1], True
                )

            else:
                # TODO: Make this cleaner - shape-1 dims are pruned so need to ignore 0 to update cache for BS=1
                curr_k = ops.dynamic_update_(past_k, last_token_indices, key, [3], [1], True)
                curr_v = ops.dynamic_update_(past_v, last_token_indices, value, [2], [1], True)

            key = curr_k
            value = curr_v

        # Repeat interleave KV heads for Grouped Query Attention
        if self.gq_groups:
            key, value = self.repeat_kv(key, value)

        attn_output = self.attention_block(query, key, value, causal_mask)
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(
            (self.config.execution.micro_batch_size * self.initial_seq_len, -1)
        )

        outputs = (attn_output,)

        if self.config.execution.use_cache:
            outputs = (
                attn_output,
                curr_k,
                curr_v,
            )

        return outputs


class LlamaSelfAttentionTP(addons.Module):
    def __init__(self, config: LlamaConfig):
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
        self.heads = LlamaAttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices
        self.output = Linear(self.config.model.hidden_size, bias=False, replica_grouping=self.replica_grouping)

    def build(
        self,
        x: popxl.Tensor,
        last_token_indices: Optional[popxl.Tensor] = None,
        past_k: Optional[popxl.Tensor] = None,
        past_v: Optional[popxl.Tensor] = None,
    ) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""

        # ----- Sharded computation -----
        h_o = self.heads(x, last_token_indices, past_k, past_v)

        z = self.output(h_o[0])

        z = replicated_all_reduce(z, group=self.replica_grouping.transpose())

        outputs = (z,)
        if self.config.execution.use_cache:
            outputs = (
                z,
                h_o[1],
                h_o[2],
            )

        return outputs

    @staticmethod
    def hf_mapping(config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        attn_tp = (
            config.execution.tensor_parallel
            if config.execution.attention_tensor_parallel is None
            else config.execution.attention_tensor_parallel
        )

        padding = variables.heads.q.weight.shape[-1] != hf_model.q_proj.weight.data.T.shape[-1] // attn_tp
        kv_padding = variables.heads.kv.weight.shape[-1] // 2 != hf_model.k_proj.weight.data.T.shape[-1] // attn_tp

        shards = (
            config.execution.tensor_parallel
            if config.execution.attention_tensor_parallel is None
            else config.execution.attention_tensor_parallel
        )

        attn_tp = (
            config.execution.tensor_parallel
            if config.execution.attention_tensor_parallel is None
            else config.execution.attention_tensor_parallel
        )

        hf_query_w = to_numpy(hf_model.q_proj.weight.data, dtype).T
        hf_key_w = to_numpy(hf_model.k_proj.weight.data, dtype).T
        hf_value_w = to_numpy(hf_model.v_proj.weight.data, dtype).T
        hf_out_proj_w = to_numpy(hf_model.o_proj.weight.data.T, dtype)

        if padding:
            hf_query_w = pad_axis(hf_query_w, variables.heads.q.weight.shape[-1] * shards, axis=1)
            hf_out_proj_w = pad_axis(hf_out_proj_w, variables.output.weight.shape[0] * shards, axis=0)

        if kv_padding:
            hf_key_w = pad_axis(hf_key_w, variables.heads.kv.weight.shape[-1] // 2 * shards, axis=1)
            hf_value_w = pad_axis(hf_value_w, variables.heads.kv.weight.shape[-1] // 2 * shards, axis=1)

        query_w = shard(hf_query_w, attn_tp, -1)
        key_w = shard(hf_key_w, attn_tp, -1)
        value_w = shard(hf_value_w, attn_tp, -1)
        out_proj_w = shard(hf_out_proj_w, attn_tp, axis=0)

        q_w = np.ascontiguousarray(np.concatenate([query_w[i][np.newaxis, ...] for i in range(attn_tp)]))

        kv_w = np.ascontiguousarray(
            np.concatenate([np.concatenate([key_w[i], value_w[i]], axis=-1)[np.newaxis, ...] for i in range(attn_tp)])
        )

        return {
            variables.heads.q.weight: q_w,
            variables.heads.kv.weight: kv_w,
            variables.output.weight: out_proj_w,
        }
