# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Optional, Dict, Tuple

import popxl
from popxl.utils import to_numpy
from popxl import ops

import popxl_addons as addons
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_grad_inputs
from popxl_addons import NamedTensors
from popxl.tensor import HostTensor

from utils.utils import shard, repeat
from config import GPTConfig

from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HFModel


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
        tp = config.execution.tensor_parallel
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
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        """
        When using sharding, if vocab_size is not a multiple of n_shards, the resulting embedding layers have alarger
        effective vocab_size wrt the hf_model. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.model.dtype
        n_shards = config.execution.tensor_parallel
        word_shard_size, pos_shard_size = GPTEmbeddingsTP.get_vocab_shard_sizes(config)

        word_pad = word_shard_size * n_shards - config.model.embedding.vocab_size
        pos_pad = pos_shard_size * n_shards - config.model.embedding.max_positional_length

        # Pad only first axis in one direction
        def pad(x, n_pad):
            return np.pad(x, ((0, n_pad), (0, 0)))

        return {
            variables.word.weight: shard(pad(to_numpy(hf_model.wte.weight.data, dtype), word_pad), n_shards, axis=0),
            variables.positional.weight: shard(
                pad(to_numpy(hf_model.wpe.weight.data, dtype), pos_pad), n_shards, axis=0
            ),
        }

    @staticmethod
    def get_offsets(config: GPTConfig) -> Tuple[np.ndarray, np.ndarray]:
        n_shards = config.execution.tensor_parallel

        word_offsets = Embedding.get_offsets(config.model.embedding.vocab_size, n_shards)
        pos_offsets = Embedding.get_offsets(config.model.embedding.max_positional_length, n_shards)

        return word_offsets, pos_offsets

    @staticmethod
    def get_vocab_shard_sizes(config: GPTConfig) -> Tuple[int, int]:
        n_shards = config.execution.tensor_parallel

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
        n_shards = config.execution.tensor_parallel
        positions = generate_positions(config).flatten()
        word_offsets, pos_offsets = GPTEmbeddingsTP.get_offsets(config)

        pos_offsetted = cls._offset_input(positions, pos_offsets, n_shards, axis)

        if words is not None:
            words_offsetted = cls._offset_input(words, word_offsets, n_shards, axis)
            return words_offsetted, pos_offsetted

        else:
            return pos_offsetted

    @staticmethod
    def offset_input(data: np.ndarray, i: int, vocab_size: int, n_shards: int):
        return data - (i * Embedding.get_vocab_shard_size(vocab_size, n_shards))
