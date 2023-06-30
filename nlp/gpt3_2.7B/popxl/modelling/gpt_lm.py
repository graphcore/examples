# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
from typing import Dict, Tuple, Callable, List, Union, Optional
from scipy.stats import truncnorm

# HF
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as HFModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


import popxl
from popxl.utils import to_numpy
from popxl import ops

from popxl_addons import NamedTensors, GraphWithNamedArgs, NamedVariableFactories
import popxl_addons as addons
from popxl_addons import NamedTensors, GraphWithNamedArgs
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_inputs
from popxl_addons.ops.cross_entropy_sharded_loss import cross_entropy_sharded_loss
from popxl_addons.layers import LayerNorm
from popxl_addons.named_replica_grouping import fill_none_group
from popxl_addons.utils import WeightsDict

from modelling.gpt_model import GPTModelTP
from modelling.embedding import GPTEmbeddingsTP
from config import GPTConfig
from utils.utils import replica_groups


class GPTLMHeadTP(addons.Module):
    def __init__(self, config: GPTConfig):
        """
        Language model head for GPT, with weights sharded along the vocab axis.
        Includes a layer norm which is normally after the decoder stack. Included here for phased execution.
        Outputs sharded logits through a linear projection.
        No bias is applied.
        In GPT, the weight matrix is tied to the word embedding weights. word_embedding needs to be provided as input
        and need to be sharded as well.
        """
        super().__init__()
        self.config = config
        self.rg_tp, _ = replica_groups(config)

        vocab_shard_size = GPTEmbeddingsTP(config).word.vocab_shard_size
        self.embedding_shape = (vocab_shard_size, config.model.hidden_size)

        # identical
        self.ln_f = LayerNorm()

    def build(self, x: popxl.Tensor, word_embedding: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        x = self.ln_f(x)
        x = replicated_all_reduce_identical_inputs(x, group=self.rg_tp)

        if not word_embedding:
            self.word_embedding = self.add_variable_input(
                "word_embedding",
                partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=self.embedding_shape),
                self.config.model.dtype,
                replica_grouping=self.rg_tp.transpose(),
            )
        else:
            self.word_embedding = word_embedding

        logits = x @ self.word_embedding.T
        return logits

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: GPT2Model) -> WeightsDict:
        dtype = config.model.dtype
        weights = WeightsDict(
            {
                variables.ln_f.weight: to_numpy(hf_model.ln_f.weight.data, dtype),
                variables.ln_f.bias: to_numpy(hf_model.ln_f.bias.data, dtype),
            }
        )

        return weights


class CrossEntropyShardedLoss(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp, self.rg_dp = replica_groups(config)

        self.word_offset_np = GPTEmbeddingsTP(config).word.offsets

    def build(self, logits: popxl.Tensor, labels: popxl.Tensor) -> popxl.Tensor:
        # logits: [b*s, vocab/tp1]
        # logits: sharded tp1

        self.word_offset = self.add_variable_input(
            "fwd.word_offset",
            iter(self.word_offset_np),
            labels.dtype,
            replica_grouping=self.rg_tp.transpose(),
            overwrite=True,
        )

        # ignore index is for 0th indexed tokens
        ignore_index = -1 * self.word_offset
        labels_offsetted = labels - self.word_offset

        # loss output: identical tp1 & tp2
        return cross_entropy_sharded_loss(
            logits, labels_offsetted, ignore_index=ignore_index, reduction="mean", replica_grouping=self.rg_tp
        )


class GPTLMHeadLossTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.lm_head = GPTLMHeadTP(self.config)
        self.loss = CrossEntropyShardedLoss(self.config)

    def build(self, x: popxl.Tensor, labels: popxl.Tensor) -> popxl.Tensor:
        # x: [b*s, h]
        # x: identical tp1, sharded tp2. labels: identical tp1 & tp2
        logits = self.lm_head(x)
        loss = self.loss(logits, labels)
        # loss:  identical tp1 & tp2
        return loss

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        return GPTLMHeadTP.hf_mapping(config, variables.lm_head, hf_model)


class GPTLMHeadModelTP(addons.Module):
    def __init__(self, config: GPTConfig):
        """Language Model for GPT2/3 model (with tied weights)."""
        super().__init__()
        self.config = config

        self.transformer = GPTModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.lm_head = GPTLMHeadTP(config)

    def build(self, input_ids: popxl.Tensor, position_ids: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        x = self.transformer(input_ids, position_ids)
        word_embedding = self.transformer.embeddings.word.weight
        x = self.lm_head(x, word_embedding)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        weights = GPTModelTP.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False)
        weights.update(GPTLMHeadTP.hf_mapping(config, variables.lm_head, hf_model.transformer))

        return weights


class GPTLMHeadModelLossTP(addons.Module):
    def __init__(self, config: GPTConfig):
        """Language Model and loss for GPT2/3 model (with tied weights)."""
        super().__init__()
        self.config = config

        self.transformer = GPTModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.head = GPTLMHeadLossTP(config)

    def build(self, input_ids: popxl.Tensor, position_ids: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        x = self.transformer(input_ids, position_ids)
        word_embedding = self.transformer.embeddings.word.weight.T
        loss = self.head(x, word_embedding)
        return loss

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> WeightsDict:
        weights = GPTModelTP.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False)
        weights.update(GPTLMHeadLossTP.hf_mapping(config, variables.head, hf_model.transformer))

        return weights


class HeadFwdBwdTiedEmb(addons.Module):
    """Layer used to merge forward and backward head layers"""

    def __init__(
        self,
        config: GPTConfig,
        fwd_graph: GraphWithNamedArgs,
        bwd_graph: GraphWithNamedArgs,
        fwd_facts: NamedVariableFactories,
        bwd_facts: NamedVariableFactories,
    ):
        super().__init__()
        self.config = config
        self.fwd_graph = fwd_graph
        self.bwd_graph = bwd_graph
        self.fwd_facts = fwd_facts
        self.bwd_facts = bwd_facts

    def build(
        self, x, labels, word_embedding, word_embedding_accum_t: popxl.TensorByRef
    ) -> Tuple[popxl.Tensor, popxl.Tensor]:
        fwd_ts = self.add_variable_inputs("fwd", self.fwd_facts)
        bwd_ts = self.add_variable_inputs("bwd", self.bwd_facts)
        fwd_ts.insert("lm_head.word_embedding", word_embedding, overwrite=True)
        bwd_ts.insert("lm_head.word_embedding", word_embedding, overwrite=True)
        bwd_ts.insert("accum.lm_head.word_embedding", word_embedding_accum_t, overwrite=True)

        fwd_info = self.fwd_graph.bind(fwd_ts).call_with_info(x, labels)
        x, *_ = fwd_info.outputs

        loss_scaling = popxl.constant(self.config.execution.loss_scaling, self.config.model.dtype)

        dx, *_ = self.bwd_graph.bind(bwd_ts).call(
            loss_scaling, args=self.bwd_graph.grad_graph_info.inputs_dict(fwd_info)
        )

        return x, dx


class HeadFwdBwd(addons.Module):
    """Layer used to merge forward and backward head layers"""

    def __init__(
        self,
        config: GPTConfig,
        fwd_graph: GraphWithNamedArgs,
        bwd_graph: GraphWithNamedArgs,
        fwd_facts: NamedVariableFactories,
        bwd_facts: NamedVariableFactories,
    ):
        super().__init__()
        self.config = config
        self.fwd_graph = fwd_graph
        self.bwd_graph = bwd_graph
        self.fwd_facts = fwd_facts
        self.bwd_facts = bwd_facts

    def build(self, x, unpadded_length, labels) -> Tuple[popxl.Tensor, popxl.Tensor]:
        fwd_ts = self.add_variable_inputs("fwd", self.fwd_facts)
        bwd_ts = self.add_variable_inputs("bwd", self.bwd_facts)

        fwd_info = self.fwd_graph.bind(fwd_ts).call_with_info(x, unpadded_length, labels)
        x, logits, *_ = fwd_info.outputs

        loss_scaling = popxl.constant(self.config.execution.loss_scaling, self.config.model.dtype)

        dx, *_ = self.bwd_graph.bind(bwd_ts).call(
            loss_scaling, args=self.bwd_graph.grad_graph_info.inputs_dict(fwd_info)
        )

        return x, dx, logits


def generate_greedy_tp(config: GPTConfig, logits: popxl.Tensor, last_token_index: popxl.Tensor):
    """
    Generate a new token based on greedy choice.
    Args:
        logits (popxl.Tensor, int32): Sharded logits for the whole sequence. Shape (seq_len, vocab_shard_size)
        last_token_index (popxl.Tensor, int32): Indices locating the last valid (non-padded) token for each batch. Logits at that indices correspond
                                                to the logits for the new token. It should be of shape (micro_batch_size,).
    Returns:
        (popxl.Tensor, int32): new token ids, of shape (micro_batch_size,)
    """
    rg_tp, _ = replica_groups(config)
    vocab_size = config.model.embedding.vocab_size
    vocab_size_sharded = logits.shape[1]
    tp = config.execution.tensor_parallel
    # indices for next token logits in each batch
    offsetted_batch_indices = popxl.constant(
        np.arange(0, config.execution.micro_batch_size) * config.model.sequence_length, dtype=popxl.int32
    )
    offsetted_batch_indices = last_token_index + offsetted_batch_indices
    # next token logits, sharded
    next_token_logits = logits[offsetted_batch_indices]  # (tp, mb_size, vocab_shard_size)

    # gather tensor parallel shards and get full logits: (mb_size, vocab_size)
    next_token_logits = ops.collectives.replicated_all_gather(next_token_logits, axis=1, group=rg_tp)
    next_token_logits = next_token_logits[:, :vocab_size]  # Remove padded vocab

    # (mb_size,)
    return ops.argmax(next_token_logits, dim=1)
