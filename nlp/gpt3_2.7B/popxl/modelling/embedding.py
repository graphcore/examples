# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Optional, Dict, Tuple
from math import ceil
from scipy.stats import truncnorm
from functools import partial
import logging

import popxl
from popxl.utils import to_numpy
from popxl import ops, ReplicaGrouping
from popxl.tensor import HostTensor


import popxl_addons as addons
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_grad_inputs
from popxl_addons import NamedTensors
from popxl_addons.utils import WeightsDict
from popxl_addons.array_munging import pad_axis, shard, repeat

from utils.utils import replica_groups
from config import GPTConfig

from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HFModel

logger = logging.getLogger(__name__)

# The dynamic slice libraries particularly like multiples of 16.
MULTISLICE_MAGIC_NUMBER = 16


def generate_positions(config: GPTConfig) -> np.ndarray:
    pos = (
        np.repeat(np.arange(0, config.model.sequence_length).reshape(1, -1), config.execution.micro_batch_size, axis=0)
        .flatten()
        .copy()
    )
    return pos


class EmbeddingTP(addons.Module):
    def __init__(
        self,
        dtype: popxl.dtype,
        vocab_size: int,
        hidden_size: int,
        rg_tp: ReplicaGrouping,
        axis: int = 0,
    ):
        """Embedding that is shared along vocab axis

        Args:
            dtype: numerical type
            vocab_size: dimension of the input space. Input indices take value in this space, ranging from 0 ... vocab_size
            hidden_size: dimension of the output space, (dimension of embedded vectors). Each input index corresponds to a distinct vector of size hidden_size.
            axis: vocab axis. Each index selects elements in the embedding matrix along this axis. Default to 0: the vocab axis is along axis 0, rows.
            rg_tp: comms replica grouping that shards the vocab axis
        """
        super().__init__()
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.axis = axis
        self.rg_tp = rg_tp

        self.vocab_shard_size = self.get_vocab_shard_size(self.vocab_size, self.rg_tp.group_size)
        self.vocab_size_padded = self.get_padded_size(self.vocab_size, self.rg_tp.group_size)
        self.offsets = self.get_offsets(self.vocab_size, self.rg_tp.group_size)

    @staticmethod
    def get_vocab_shard_size(vocab_size: int, n_shards: int) -> int:
        """Vocab size per shard."""
        shard_size = ceil(vocab_size / n_shards)
        return MULTISLICE_MAGIC_NUMBER * ceil(shard_size / MULTISLICE_MAGIC_NUMBER)

    @staticmethod
    def get_padded_size(vocab_size: int, n_shards: int) -> int:
        """Total size of vocab including padding. vocab_size should be unsharded."""
        vocab_shard_size = EmbeddingTP.get_vocab_shard_size(vocab_size, n_shards)
        return n_shards * vocab_shard_size

    @staticmethod
    def get_offsets(vocab_size: int, n_shards: int) -> np.ndarray:
        """Indices offset per vocab shard. vocab_size should be unsharded."""
        vocab_shard_size = EmbeddingTP.get_vocab_shard_size(vocab_size, n_shards)
        return np.arange(n_shards * vocab_shard_size, step=vocab_shard_size)

    def build(self, indices: popxl.Tensor) -> popxl.Tensor:
        """

        Args:
            indices: token indices. Shape (..., sequence_length). Value range: [0, vocab_size)

        Returns:
            Embedded vectors for each index. Shape (...,sequence_length, hidden_size)
            Embedding corresponds to a table lookup: each index in `indices` selects a row in the embedding weights matrix.
            If using sharding, out of range indices will be automatically set to zero.
        """
        # indices: shape: [b*s] shard: [DP]
        self.offset = self.add_variable_input(
            "offset",
            iter(self.offsets),
            indices.dtype,
            replica_grouping=self.rg_tp.transpose(),
        )

        indices_offsetted = indices - self.offset

        self.weight = self.add_variable_input(
            "weight",
            partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=(self.vocab_shard_size, self.hidden_size)),
            self.dtype,
            replica_grouping=self.rg_tp.transpose(),
        )

        # output: shape: [b*s,h], shard [DP,TP]
        return ops.gather(self.weight, indices_offsetted, axis=self.axis, zero_OOR=True)


class GPTEmbeddingsTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp, _ = replica_groups(config)

        self.word = EmbeddingTP(
            self.config.model.dtype,
            self.config.model.embedding.vocab_size,
            self.config.model.hidden_size,
            rg_tp=self.rg_tp,
        )
        self.positional = EmbeddingTP(
            self.config.model.dtype,
            self.config.model.embedding.max_positional_length,
            self.config.model.hidden_size,
            rg_tp=self.rg_tp,
        )

    def build(
        self, input_ids: popxl.Tensor, position_ids: popxl.Tensor, seed: Optional[popxl.Tensor] = None
    ) -> popxl.Tensor:
        # input_ids/position_ids: shape: [b*s]. identical TP
        # seed: identical TP, different DP
        input_ids = input_ids.flatten()
        position_ids = position_ids.flatten()

        words_embeds = self.word(input_ids)
        position_embeds = self.positional(position_ids)
        x = words_embeds + position_embeds
        x = replicated_all_reduce_identical_grad_inputs(x, group=self.rg_tp)

        # Identical computation
        if not self.config.model.eval:
            assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
            x = ops.dropout(x, seed, p=self.config.model.dropout_prob)

        # output: identical TP
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        """
        When using sharding, if vocab_size is not a multiple of n_shards, the resulting embedding layers have a larger
        effective vocab_size wrt the hf_model. In that case, extra zero-padded elements are added to the embedding
        matrix along the vocab axis BEFORE sharding the hf weight matrix (all padding is located at OOR indices)
        """
        dtype = config.model.dtype
        tp = config.execution.tensor_parallel

        vocab_size = config.model.embedding.vocab_size
        max_pos_len = config.model.embedding.max_positional_length

        wte = hf_model.wte.weight.data
        wpe = hf_model.wpe.weight.data

        assert wte.shape[0] <= vocab_size, f"{wte.shape[0]} != {vocab_size}"
        assert wpe.shape[0] <= max_pos_len, f"{wpe.shape[0]} != {max_pos_len}"

        # TODO: loss 80 due to padded word embeddings and magic number
        # padding doesn't seem to work
        # if wpe.shape[0] > max_pos_len:
        #     logger.warning(
        #         "HuggingFace model has a longer positional embedding matrix than the max_positional_length of the model. "
        #         f"The matrix will be trimmed. {wpe.shape[0]} > {max_pos_len}"
        #     )
        #     wpe = wpe[:max_pos_len]

        word_padded = EmbeddingTP.get_padded_size(vocab_size, tp)
        pos_padded = EmbeddingTP.get_padded_size(max_pos_len, tp)

        return WeightsDict(
            {
                variables.word.weight: shard(pad_axis(to_numpy(wte, dtype), word_padded, axis=0), tp, axis=0),
                variables.positional.weight: shard(pad_axis(to_numpy(wpe, dtype), pos_padded, axis=0), tp, axis=0),
            }
        )
