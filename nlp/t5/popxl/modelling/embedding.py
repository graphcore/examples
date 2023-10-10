# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import math
import torch
from functools import partial
from scipy.stats import truncnorm
from typing import Optional, Dict

import popxl
from popxl.utils import to_numpy
from popxl import ops, ReplicaGrouping

import popxl_addons as addons
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons import NamedTensors
from popxl_addons.named_tensors import NamedTensorData
from popxl_addons.array_munging import repeat, shard
from popxl.tensor import HostTensor

from config import T5Config

from transformers.models.t5.modeling_t5 import T5Model as HFModel
from transformers.models.t5.configuration_t5 import T5Config as T5ConfigHF


class T5EmbeddingsTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        self.word = Embedding(
            self.config.model.dtype,
            self.config.model.embedding.vocab_size,
            self.config.model.hidden_size,
            replica_grouping=self.replica_grouping,
        )
        # Relative position embeddings
        n_heads_groups = self.replica_grouping.num_groups
        self.n_heads = self.config.model.attention.heads // n_heads_groups
        self.relative_attention_num_buckets = self.config.model.attention.relative_attention_num_buckets
        self.should_upcast = config.model.scale_ff > 1

    def build(self, input_ids: popxl.Tensor, seed: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        """`input_ids` are offsetted. Identical outputs across shards"""
        input_ids = input_ids.flatten()
        x = self.word(input_ids)

        x = replicated_all_reduce_identical_grad_inputs(x, group=self.replica_grouping.transpose())

        # Identical computation
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)

        # Relative position embeddings: independent on each device
        # Just create the weight here, but it will be used by the encoder layers
        self.rel_pos_weight = self.add_variable_input(
            "rel_pos_weight",
            partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=(self.relative_attention_num_buckets, self.n_heads)),
            self.config.model.dtype,
            replica_grouping=self.replica_grouping,
        )

        if self.should_upcast:
            x = ops.cast(x, popxl.float32)
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        """
        When using sharding, if vocab_size is not a multiple of n_shards, the resulting embedding layers have a larger
        effective vocab_size wrt the hf_model. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel
        word_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, n_shards)
        word_pad = word_shard_size * n_shards - config.model.embedding.vocab_size
        # Pad only first axis in one direction
        def pad(x, n_pad):
            return np.pad(x, ((0, n_pad), (0, 0)))

        # Relative position embedding weights
        rel_pos_w = to_numpy(
            hf_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.data, dtype
        ).T

        return {
            variables.word.weight: shard(pad(to_numpy(hf_model.shared.weight.data, dtype), word_pad), n_shards, axis=0),
            variables.rel_pos_weight: shard(rel_pos_w, n_shards, axis=0).transpose((0, 2, 1)),
        }

    @staticmethod
    def to_hf(config: T5ConfigHF, variables_data: NamedTensorData, hf_model: HFModel) -> Dict[str, torch.Tensor]:
        state_dict = {
            "shared.weight": torch.tensor(
                np.concatenate(variables_data.word.weight, axis=0)[: config.vocab_size], dtype=config.torch_dtype
            )
        }
        return state_dict

    @staticmethod
    def get_offsets(config: T5Config) -> np.ndarray:
        n_shards = config.execution.tensor_parallel

        word_offsets = Embedding.get_offsets(config.model.embedding.vocab_size, n_shards)
        return word_offsets

    @staticmethod
    def get_vocab_shard_sizes(config: T5Config) -> int:
        n_shards = config.execution.tensor_parallel

        word_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, n_shards)

        return word_shard_size

    @staticmethod
    def _offset_input(data: HostTensor, offsets: HostTensor, n_shards, axis: int = 0):
        def bc_shape(t):
            # Shape for broadcasting. `slice(None, None)` represents all like `array[:]`
            shape = [np.newaxis] * len(t.shape)
            shape.insert(axis, slice(None, None))
            return tuple(shape)

        data_offsetted = repeat(data, n_shards, axis) - offsets[bc_shape(data)]
        return data_offsetted

    @classmethod
    def offset_inputs(cls, config: T5Config, words: HostTensor, axis: int = 0):
        n_shards = config.execution.tensor_parallel
        word_offsets = T5EmbeddingsTP.get_offsets(config)

        words_offsetted = cls._offset_input(words, word_offsets, n_shards, axis)
        return words_offsetted

    @staticmethod
    def offset_input(data: np.ndarray, i: int, config: T5Config):
        return data - (
            i * Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, config.execution.tensor_parallel)
        )


class T5DecoderEmbeddingsTP(addons.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # Relative position embeddings
        n_heads_groups = self.replica_grouping.num_groups
        self.n_heads = self.config.model.attention.heads // n_heads_groups
        self.relative_attention_num_buckets = self.config.model.attention.relative_attention_num_buckets
        self.should_upcast = config.model.scale_ff > 1

    def build(
        self, input_ids: popxl.Tensor, word_embedding: popxl.Tensor, seed: Optional[popxl.Tensor] = None
    ) -> popxl.Tensor:
        """`input_ids` are offsetted. Identical outputs across shards"""
        input_ids = input_ids.flatten()

        # Perform an embedding lookup operation using the given weights
        x = ops.gather(word_embedding, input_ids, zero_OOR=self.replica_grouping.num_groups > 1)

        x = replicated_all_reduce_identical_grad_inputs(x, group=self.replica_grouping.transpose())

        # Identical computation
        if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)

        # Relative position embeddings: independent on each device
        # Just create the weight here, but it will be used by the decoder layers
        self.rel_pos_weight = self.add_variable_input(
            "rel_pos_weight",
            partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=(self.relative_attention_num_buckets, self.n_heads)),
            self.config.model.dtype,
            replica_grouping=self.replica_grouping,
        )

        if self.should_upcast:
            x = ops.cast(x, popxl.float32)
        return x

    @staticmethod
    def hf_mapping(config: T5Config, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        # The only variable owned by this layer is the relative position embedding
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel
        rel_pos_w = to_numpy(
            hf_model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.data, dtype
        ).T

        return {variables.rel_pos_weight: shard(rel_pos_w, n_shards, axis=0).transpose((0, 2, 1))}


class T5RelPosEmbeddingsTP(addons.Module):
    def __init__(
        self, dtype: popxl.dtype, vocab_size: int, hidden_size: int, replica_grouping: Optional[ReplicaGrouping] = None
    ):
        super().__init__()
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.replica_grouping = replica_grouping

    def build(self, input_ids: popxl.Tensor, rel_pos_weight: popxl.Tensor = None) -> popxl.Tensor:
        """An embedding lookup, where the weight is TP-sharded not across the vocab axis, but across the output axis."""
        if rel_pos_weight is None:
            # Create the weight if it is not given
            self.weight = self.add_variable_input(
                "weight",
                partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=(self.vocab_size, self.hidden_size)),
                self.dtype,
                replica_grouping=self.replica_grouping,
            )
            rel_pos_weight = self.weight

        rel_pos_encodings = ops.gather(rel_pos_weight, input_ids)
        return rel_pos_encodings
