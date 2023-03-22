# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from math import ceil

import numpy as np
from typing import Optional, Dict, Tuple

import popxl
from popxl.utils import to_numpy
from popxl import ops, ReplicaGrouping
from scipy.stats import truncnorm

import popxl_addons as addons
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_grad_inputs
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict
from popxl.tensor import HostTensor

from popxl_addons.array_munging import shard, repeat, shard2D, pad_axis
from config import GPTConfig

from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HFModel

from utils.utils import tp2d_replica_groups

# The dynamic slice libraries particularly like multiples of 16.
MULTISLICE_MAGIC_NUMBER = 16


def generate_positions(config: GPTConfig) -> np.ndarray:
    pos = (
        np.repeat(np.arange(0, config.model.sequence_length).reshape(1, -1), config.execution.micro_batch_size, axis=0)
        .flatten()
        .copy()
    )
    return pos


class GPTEmbeddingsTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel_1
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        self.word = Embedding(
            self.config.model.dtype,
            self.config.model.embedding.vocab_size,
            self.config.model.hidden_size,
            replica_grouping=self.replica_grouping,
        )
        self.positional = Embedding(
            self.config.model.dtype,
            self.config.model.embedding.max_positional_length,
            self.config.model.hidden_size,
            replica_grouping=self.replica_grouping,
        )

    def build(
        self, input_ids: popxl.Tensor, position_ids: popxl.Tensor, seed: Optional[popxl.Tensor] = None
    ) -> popxl.Tensor:
        """`words`, `positions` are offsetted. Identical outputs across shards"""
        input_ids = input_ids.flatten()
        position_ids = position_ids.flatten()

        words_embeds = self.word(input_ids)
        position_embeds = self.positional(position_ids)
        x = words_embeds + position_embeds
        x = replicated_all_reduce_identical_grad_inputs(x, group=self.replica_grouping.transpose())

        # Identical computation
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)

        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        """
        When using sharding, if vocab_size is not a multiple of n_shards, the resulting embedding layers have a larger
        effective vocab_size wrt the hf_model. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel_1
        word_shard_size, pos_shard_size = GPTEmbeddingsTP.get_vocab_shard_sizes(config)

        word_pad = word_shard_size * n_shards - config.model.embedding.vocab_size
        pos_pad = pos_shard_size * n_shards - config.model.embedding.max_positional_length

        assert hf_model.wte.weight.data.shape[0] == config.model.embedding.vocab_size
        assert hf_model.wpe.weight.data.shape[0] == config.model.embedding.max_positional_length

        # Pad only first axis in one direction
        def pad(x, n_pad):
            return np.pad(x, ((0, n_pad), (0, 0)))

        return WeightsDict(
            {
                variables.word.weight: shard(
                    pad(to_numpy(hf_model.wte.weight.data, dtype), word_pad), n_shards, axis=0
                ),
                variables.positional.weight: shard(
                    pad(to_numpy(hf_model.wpe.weight.data, dtype), pos_pad), n_shards, axis=0
                ),
            }
        )

    @staticmethod
    def get_offsets(config: GPTConfig) -> Tuple[np.ndarray, np.ndarray]:
        n_shards = config.execution.tensor_parallel_1

        word_offsets = Embedding.get_offsets(config.model.embedding.vocab_size, n_shards)
        pos_offsets = Embedding.get_offsets(config.model.embedding.max_positional_length, n_shards)

        return word_offsets, pos_offsets

    @staticmethod
    def get_vocab_shard_sizes(config: GPTConfig) -> Tuple[int, int]:
        n_shards = config.execution.tensor_parallel_1

        word_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, n_shards)
        pos_shard_size = Embedding.get_vocab_shard_size(config.model.embedding.max_positional_length, n_shards)

        return word_shard_size, pos_shard_size

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
    def offset_inputs(
        cls,
        config: GPTConfig,
        words: Optional[HostTensor] = None,
        token_types: Optional[HostTensor] = None,
        axis: int = 0,
    ):
        n_shards = config.execution.tensor_parallel_1
        positions = generate_positions(config).flatten()
        word_offsets, pos_offsets = GPTEmbeddingsTP.get_offsets(config)

        pos_offsetted = cls._offset_input(positions, pos_offsets, n_shards, axis)

        if words is not None:
            words_offsetted = cls._offset_input(words, word_offsets, n_shards, axis)
            return words_offsetted, pos_offsetted

        else:
            return pos_offsetted

    @staticmethod
    def offset_input(data: np.ndarray, i: int, config: GPTConfig):
        return data - (
            i * Embedding.get_vocab_shard_size(config.model.embedding.vocab_size, config.execution.tensor_parallel_1)
        )


class EmbeddingTP2D(addons.Module):
    def __init__(
        self,
        dtype: popxl.dtype,
        vocab_size: int,
        hidden_size: int,
        rg_tp1: ReplicaGrouping,
        rg_tp2: ReplicaGrouping,
        rg_tp_all: ReplicaGrouping,
        axis: int = 0,
    ):
        """
        Args:
            dtype: numerical type
            vocab_size: dimension of the input space. Input indices take value in this space, ranging from 0 ... vocab_size
            hidden_size: dimension of the output space, (dimension of embedded vectors). Each input index corresponds to a distinct vector of size hidden_size.
            axis: vocab axis. Each index selects elements in the embedding matrix along this axis. Default to 0: the vocab axis is along axis 0, rows.
            rg_tp1: comms replica grouping that shards the vocab axis
            rg_tp2: comms replica grouping that shards the hidden axis
            rg_tp_all: comms replica grouping for all TP shards (tensor product of tp1 and tp2)
        """
        super().__init__()
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.axis = axis
        self.rg_tp1 = rg_tp1
        self.rg_tp2 = rg_tp2
        self.rg_tp_all = rg_tp_all

        assert self.hidden_size % self.rg_tp2.group_size == 0

        self.vocab_shard_size = self.get_vocab_shard_size(self.vocab_size, self.rg_tp1.group_size)
        self.hidden_shard_size = self.hidden_size // self.rg_tp2.group_size
        self.offsets = self.get_offsets(self.vocab_size, self.rg_tp1.group_size)

    @staticmethod
    def get_vocab_shard_size(vocab_size: int, n_shards: int) -> int:
        """Vocab size per shard."""
        shard_size = ceil(vocab_size / n_shards)
        return MULTISLICE_MAGIC_NUMBER * ceil(shard_size / MULTISLICE_MAGIC_NUMBER)

    @staticmethod
    def get_padded_size(vocab_size: int, n_shards: int) -> int:
        """Total size of vocab including padding. vocab_size should be unsharded."""
        vocab_shard_size = Embedding.get_vocab_shard_size(vocab_size, n_shards)
        return n_shards * vocab_shard_size

    @staticmethod
    def get_offsets(vocab_size: int, n_shards: int) -> np.ndarray:
        """Indices offset per vocab shard. vocab_size should be unsharded."""
        vocab_shard_size = Embedding.get_vocab_shard_size(vocab_size, n_shards)
        return np.arange(n_shards * vocab_shard_size, step=vocab_shard_size)

    def build(self, indices: popxl.Tensor) -> popxl.Tensor:
        """

        Args:
            indices: token indices. Shape (...,sequence_length, vocab_size)

        Returns:
            Embedded vectors for each index. Shape (...,sequence_length, hidden_size)
            Embedding corresponds to a table lookup: each index in `indices` selects a row in the embedding weights matrix.
            If using sharding, out of range indices will be automatically set to zero.
        """
        # indices: identical tp1 & tp2
        self.offset = self.add_variable_input(
            "offset",
            iter(self.offsets),
            indices.dtype,
            replica_grouping=self.rg_tp1.transpose(),
        )

        indices_offsetted = indices - self.offset

        self.weight = self.add_variable_input(
            "weight",
            partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=(self.vocab_shard_size, self.hidden_shard_size)),
            self.dtype,
            replica_grouping=self.rg_tp_all.transpose(),
        )

        return ops.gather(self.weight, indices_offsetted, axis=self.axis, zero_OOR=True)


class GPTEmbeddingsTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        self.word = EmbeddingTP2D(
            self.config.model.dtype,
            self.config.model.embedding.vocab_size,
            self.config.model.hidden_size,
            self.rg_tp1,
            self.rg_tp2,
            self.rg_tp_all,
        )

        self.positional = EmbeddingTP2D(
            self.config.model.dtype,
            self.config.model.embedding.max_positional_length,
            self.config.model.hidden_size,
            self.rg_tp1,
            self.rg_tp2,
            self.rg_tp_all,
        )

    def build(
        self, input_ids: popxl.Tensor, position_ids: popxl.Tensor, seed: Optional[popxl.Tensor] = None
    ) -> popxl.Tensor:
        """`words`, `positions` are offsetted. Identical outputs across shards"""
        # input_ids, position_ids: identical tp1 & tp2
        input_ids = input_ids.flatten()
        position_ids = position_ids.flatten()

        words_embeds = self.word(input_ids)
        position_embeds = self.positional(position_ids)
        x = words_embeds + position_embeds
        x = replicated_all_reduce_identical_grad_inputs(x, group=self.rg_tp1)

        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)

        # x: identical tp1. sharded tp2
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        """
        When using sharding, if vocab_size is not a multiple of n_shards, the resulting embedding layers have a larger
        effective vocab_size wrt the hf_model. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.model.dtype
        tp1 = config.execution.tensor_parallel_1
        tp2 = config.execution.tensor_parallel_2

        vocab_size = config.model.embedding.vocab_size
        seq_len = config.model.embedding.max_positional_length

        assert hf_model.wte.weight.data.shape[0] == vocab_size
        assert hf_model.wpe.weight.data.shape[0] == seq_len

        word_padded = EmbeddingTP2D.get_padded_size(vocab_size, tp1)
        pos_padded = EmbeddingTP2D.get_padded_size(seq_len, tp1)

        return WeightsDict(
            {
                variables.word.weight: shard2D(
                    pad_axis(to_numpy(hf_model.wte.weight.data, dtype), word_padded, 0), tp1, tp2, 0, 1
                ),
                variables.positional.weight: shard2D(
                    pad_axis(to_numpy(hf_model.wpe.weight.data, dtype), pos_padded, 0), tp1, tp2, 0, 1
                ),
            }
        )
