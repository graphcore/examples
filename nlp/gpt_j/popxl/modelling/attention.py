# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict
import logging
import math
import torch

import popxl
from popxl import ops, ReplicaGrouping
from popxl.utils import to_numpy
from typing import Optional

import popxl_addons as addons
from popxl_addons import NamedTensors
from popxl_addons.named_tensors import NamedTensorData
from utils.utils import shard
from popxl_addons.layers import Linear
from popxl_addons.layers.linear_gq import LinearGQ, group_quantize_compress_numpy

from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons.ops.rotary_pos_embed import rotary_pos_embed, trig_table_constants

from config import GPTJConfig
from transformers.models.gptj.modeling_gptj import GPTJAttention as HFModel
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTJConfigHF


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
            self.config.model.attention.heads % n_heads_groups == 0
        ), f"{self.config.model.attention.heads} % {n_heads_groups} != 0"

        self.n_heads_groups = n_heads_groups
        self.n_heads = self.config.model.attention.heads // n_heads_groups

        # Setup
        layer_kwargs = {}
        if config.execution.group_quantise_weights > 0:
            layer = LinearGQ
            layer_kwargs["group_size"] = config.execution.group_quantise_weights
            layer_kwargs["dim"] = config.execution.group_quantise_dim
        else:
            layer = Linear

        self.qkv = layer(
            3 * self.config.model.hidden_size // n_heads_groups,
            replica_grouping=replica_grouping,
            bias=False,
            **layer_kwargs,
        )
        self.rotary_dim = self.config.model.attention.rotary_dim or self.config.model.hidden_size // self.n_heads

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None):
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
        # Optim: outline below?
        query = rotary_pos_embed(query, sin, cos, self.rotary_dim).transpose((0, 2, 1, 3))
        key = rotary_pos_embed(key, sin, cos, self.rotary_dim).transpose((0, 2, 3, 1))
        value = value.transpose((0, 2, 1, 3))

        causal_mask = popxl.constant(
            # HF uses 1e9 which is beyond fp16 range
            1e4
            * (
                np.tril(
                    np.ones(
                        (
                            self.config.model.sequence_length,
                            self.config.model.sequence_length,
                        )
                    )
                )
                - 1
            ),
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
            (
                self.config.execution.micro_batch_size * self.config.model.sequence_length,
                -1,
            )
        )

    def attention_block(
        self,
        query: popxl.Tensor,
        key: popxl.Tensor,
        value: popxl.Tensor,
        mask: popxl.Tensor,
        seed: popxl.Tensor,
    ):
        attn_weights = query @ key

        attn_weights = attn_weights * (1 / math.sqrt(value.shape[-1]))
        attn_weights = attn_weights + mask

        attn_scores = ops.softmax(attn_weights, axis=-1)
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            attn_scores = ops.dropout(attn_scores, seed, p=self.config.model.dropout_prob)

        return attn_scores @ value

    @staticmethod
    def hf_mapping(config: GPTJConfig, vars: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        query_w = to_numpy(hf_model.q_proj.weight.data, dtype).T
        key_w = to_numpy(hf_model.k_proj.weight.data, dtype).T
        value_w = to_numpy(hf_model.v_proj.weight.data, dtype).T

        return {vars.qkv.weight: np.ascontiguousarray(np.concatenate((query_w, key_w, value_w), axis=-1))}


class GPTJSelfAttentionTP(addons.Module):
    def __init__(self, config: GPTJConfig):
        super().__init__()

        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

        # Sharded across devices
        self.heads = GPTJAttentionHeads(config=config, replica_grouping=self.replica_grouping)

        # Sharded across devices
        # Setup
        layer_kwargs = {}
        if config.execution.group_quantise_weights > 0:
            layer = LinearGQ
            layer_kwargs["group_size"] = config.execution.group_quantise_weights
            layer_kwargs["dim"] = config.execution.group_quantise_dim
        else:
            layer = Linear
        self.output = layer(
            self.config.model.hidden_size,
            bias=False,
            replica_grouping=self.replica_grouping,
            **layer_kwargs,
        )

    def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        """Identical inputs and identical outputs across shards"""
        heads_seed = None
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            seed, heads_seed = ops.split_random_seed(seed)

        # ----- Identical computation -----
        z = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())

        # ----- Sharded computation -----
        z = self.heads(z, seed=heads_seed)
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

        hf_query_w = to_numpy(hf_model.q_proj.weight.data, dtype).T
        hf_key_w = to_numpy(hf_model.k_proj.weight.data, dtype).T
        hf_value_w = to_numpy(hf_model.v_proj.weight.data, dtype).T

        query_w = shard(hf_query_w, n_shards, -1)
        key_w = shard(hf_key_w, n_shards, -1)
        value_w = shard(hf_value_w, n_shards, axis=-1)

        qkv_w = np.ascontiguousarray(
            np.concatenate(
                [np.concatenate([query_w[i], key_w[i], value_w[i]], axis=-1)[np.newaxis, ...] for i in range(n_shards)]
            )
        )

        out_proj_w = to_numpy(hf_model.out_proj.weight.data.T, dtype)
        out_proj_w = shard(out_proj_w, n_shards, axis=0)

        if config.execution.group_quantise_weights:
            if config.execution.group_quantise_dim != -1:
                # shift group_quantise_dim to account for TP sharding dim
                gqdim = config.execution.group_quantise_dim + 1
            else:
                gqdim = -1
            layer_name = variables.heads.qkv.weight_compressed.name.replace(".heads.qkv.weight_compressed", "")
            logging.info(f"Quantizing {layer_name} weights")
            qkv_w = group_quantize_compress_numpy(
                qkv_w,
                config.execution.group_quantise_weights,
                gqdim,
            )
            out_proj_w = group_quantize_compress_numpy(
                out_proj_w,
                config.execution.group_quantise_weights,
                gqdim,
            )
            heads_state_dict = dict(
                zip(
                    (
                        variables.heads.qkv.weight_compressed,
                        variables.heads.qkv.weight_decompression_scale,
                        variables.heads.qkv.weight_decompression_bias,
                    ),
                    qkv_w,
                ),
            )
            output_state_dict = dict(
                zip(
                    (
                        variables.output.weight_compressed,
                        variables.output.weight_decompression_scale,
                        variables.output.weight_decompression_bias,
                    ),
                    out_proj_w,
                )
            )
            state_dict = {**heads_state_dict, **output_state_dict}
            return state_dict

        else:
            return {
                variables.heads.qkv.weight: qkv_w,
                variables.output.weight: out_proj_w,
            }

    @staticmethod
    def to_hf(config: GPTJConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        q, k, v = np.split(variables_data.heads.qkv.weight, 3, axis=-1)

        # ignored keys in HF model: masked_bias and bias
        # bias is the attention mask
        # masked bias is the constant used to translate to infinity (1e4)
        state_dict = {}
        state_dict["q_proj.weight"] = torch.tensor(
            np.concatenate(q.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype
        )
        state_dict["k_proj.weight"] = torch.tensor(
            np.concatenate(k.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype
        )
        state_dict["v_proj.weight"] = torch.tensor(
            np.concatenate(v.transpose((0, 2, 1)), axis=0), dtype=config.torch_dtype
        )
        state_dict["out_proj.weight"] = torch.tensor(
            np.concatenate(variables_data.output.weight, axis=0).T,
            dtype=config.torch_dtype,
        )

        return state_dict
