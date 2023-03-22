# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from scipy.stats import truncnorm
from typing import Optional, Tuple

# HF
from transformers.models.gpt2 import GPT2LMHeadModel as HFGPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from modelling.embedding import EmbeddingTP2D, GPTEmbeddingsTP2D
from popxl_addons import NamedTensors, GraphWithNamedArgs, NamedVariableFactories
from popxl_addons.array_munging import shard
from popxl_addons.layers import Embedding
from popxl_addons.layers.layer_norm_distributed import LayerNormDistributed
from popxl_addons.ops.replicated_all_reduce_TP import (
    replicated_all_reduce_identical_inputs,
    replicated_all_reduce_identical_grad_inputs,
)
from popxl_addons.ops.cross_entropy_sharded_loss import cross_entropy_sharded_loss
from popxl_addons.layers import LayerNorm
from popxl_addons.utils import WeightsDict

from modelling.gpt_model import GPTModelTP, GPTModelTP2D
from config import GPTConfig
from utils.utils import tp2d_replica_groups


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
        tp = config.execution.tensor_parallel_1
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # identical
        self.ln_f = LayerNorm()

    def build(self, x: popxl.Tensor, word_embedding: popxl.Tensor) -> popxl.Tensor:
        x = self.ln_f(x)
        x = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())
        logits = x @ word_embedding
        return logits

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: GPT2Model) -> WeightsDict:
        dtype = config.model.dtype
        weights = LayerNorm.torch_mapping(variables.ln_f, hf_model.ln_f, dtype)
        return weights


class GPTLMHeadModelTP(addons.Module):
    def __init__(self, config: GPTConfig):
        """GPT model (GPT2-GPT3 architecture) with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = GPTModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.lm_head = GPTLMHeadTP(config)

    def build(self, input_ids: popxl.Tensor, position_ids: Optional[popxl.Tensor] = None) -> popxl.Tensor:

        x = self.transformer(input_ids, position_ids)
        word_embedding = self.transformer.embeddings.word.weight.T
        x = self.lm_head(x, word_embedding)

        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: GPT2Model) -> WeightsDict:
        dtype = config.model.dtype
        weights = GPTModelTP.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False)
        weights.update(GPTLMHeadTP.hf_mapping(config, variables.lm_head, hf_model.transformer))

        return weights


class GPTLMHeadLossAndGradTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel_1
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

    def build(
        self,
        x: popxl.Tensor,
        labels: popxl.Tensor,
        word_embedding_t: popxl.Tensor,
        word_embedding_accum_t: popxl.TensorByRef,
        word_offset: popxl.Tensor,
    ) -> Tuple[popxl.Tensor, popxl.Tensor]:
        ir = popxl.gcg().ir

        vocab_shard_size = Embedding.get_vocab_shard_size(
            self.config.model.embedding.vocab_size, self.config.execution.tensor_parallel_1
        )
        word_embed = popxl.TensorSpec((self.config.model.hidden_size, vocab_shard_size), dtype=x.dtype)

        fwd_facts, fwd_graph = GPTLMHeadTP(self.config).create_graph(x, word_embed)
        ts = self.add_variable_inputs("fwd", fwd_facts)

        ignore_index = -1 * word_offset
        loss_graph = GraphWithNamedArgs(
            ir.create_graph(
                cross_entropy_sharded_loss,
                fwd_graph.graph.outputs[0],
                labels,
                ignore_index=ignore_index,
                reduction="mean",
                replica_grouping=self.replica_grouping.transpose(),
            )
        )

        required_grads = [fwd_graph.graph.inputs[0]]
        accums = list(fwd_graph.args.tensors) + [fwd_graph.graph.inputs[1]]  # layer norm weights + tied weight
        fwd_facts.replica_groupings.insert("word_embedding", self.replica_grouping)

        bwd_facts, bwd_graph = addons.transforms.autodiff_with_accumulation(
            fwd_graph,
            tensors_to_accumulate_grads=accums,
            grads_required=required_grads,
            replica_groupings=fwd_facts.replica_groupings,
        )
        loss_bwd = addons.transforms.autodiff(loss_graph, grads_required=(loss_graph.graph.inputs[0],))

        fwd_info = fwd_graph.bind(ts).call_with_info(x, word_embedding_t)

        logits = fwd_info.parent_output(0)
        loss_fwd_info = loss_graph.call_with_info(logits, labels, ignore_index)
        loss = loss_fwd_info.parent_output(0)

        loss_scaling = popxl.constant(self.config.execution.loss_scaling, self.config.model.dtype)

        (dx,) = loss_bwd.call(loss_scaling, args=loss_bwd.grad_graph_info.inputs_dict(loss_fwd_info))

        ln_facts = bwd_facts.copy()
        ln_facts.accum.pop("word_embedding")
        bwd_weights = self.add_variable_inputs("bwd", ln_facts)

        input_dict = bwd_graph.grad_graph_info.inputs_dict(fwd_info)
        input_dict.update({bwd_graph.args.accum.word_embedding: word_embedding_accum_t})

        (dx,) = bwd_graph.bind(bwd_weights).call(dx, args=input_dict)

        return loss, dx


class GPTLMHeadTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        embedding = GPTEmbeddingsTP2D(config)
        self.embedding_shape = (embedding.word.vocab_shard_size, embedding.word.hidden_shard_size)

        self.ln_f = LayerNormDistributed(self.rg_tp2)

    def build(self, x: popxl.Tensor, word_embedding: Optional[popxl.Tensor] = None) -> popxl.Tensor:
        # x: [b*s, h/tp2]
        # x: identical tp1, sharded tp2

        x = self.ln_f(x)
        x = replicated_all_reduce_identical_inputs(x, group=self.rg_tp1)
        # x: identical tp1, sharded tp2

        if not word_embedding:
            self.word_embedding = self.add_variable_input(
                "word_embedding",
                partial(truncnorm.rvs, -2, 2, loc=0, scale=0.02, size=self.embedding_shape),
                self.config.model.dtype,
                replica_grouping=self.rg_tp_all.transpose(),
            )
        else:
            self.word_embedding = word_embedding

        logits = x @ self.word_embedding.T
        logits = replicated_all_reduce_identical_grad_inputs(logits, group=self.rg_tp2)

        # logits: [b*s, vocab/tp1]
        # logits: sharded tp1, identical tp2
        return logits

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: GPT2Model) -> WeightsDict:
        dtype = config.model.dtype
        tp2 = config.execution.tensor_parallel_2
        weights = WeightsDict(
            {
                variables.ln_f.weight: shard(to_numpy(hf_model.ln_f.weight.data, dtype), tp2, 0),
                variables.ln_f.bias: shard(to_numpy(hf_model.ln_f.bias.data, dtype), tp2, 0),
            }
        )

        return weights


class CrossEntropyShardedLoss(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        embedding = GPTEmbeddingsTP2D(config)
        self.word_offset_np = embedding.word.offsets

    def build(self, logits: popxl.Tensor, labels: popxl.Tensor) -> popxl.Tensor:
        # logits: [b*s, vocab/tp1]
        # logits: sharded tp1, sharded tp2

        self.word_offset = self.add_variable_input(
            "fwd.word_offset",
            iter(self.word_offset_np),
            labels.dtype,
            replica_grouping=self.rg_tp1.transpose(),
            overwrite=True,
        )

        # ignore index is for 0th indexed tokens
        ignore_index = -1 * self.word_offset
        labels_offsetted = labels - self.word_offset

        # loss output: identical tp1 & tp2
        return cross_entropy_sharded_loss(
            logits, labels_offsetted, ignore_index=ignore_index, reduction="mean", replica_grouping=self.rg_tp1
        )


class GPTLMHeadLossTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.rg_tp1, self.rg_tp2, self.rg_tp_all, _ = tp2d_replica_groups(config)

        self.head = GPTLMHeadTP2D(self.config)
        self.loss = CrossEntropyShardedLoss(self.config)

    def build(self, x: popxl.Tensor, labels: popxl.Tensor) -> popxl.Tensor:
        # x: [b*s, h/tp2]
        # x: identical tp1, sharded tp2. labels: identical tp1 & tp2
        logits = self.head(x)
        loss = self.loss(logits, labels)
        # loss:  identical tp1 & tp2
        return loss


class HeadFwdBwd(addons.Module):
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
        fwd_ts.insert("head.word_embedding", word_embedding, overwrite=True)
        bwd_ts.insert("head.word_embedding", word_embedding, overwrite=True)
        bwd_ts.insert("accum.head.word_embedding", word_embedding_accum_t, overwrite=True)

        fwd_info = self.fwd_graph.bind(fwd_ts).call_with_info(x, labels)
        x, *_ = fwd_info.outputs

        loss_scaling = popxl.constant(self.config.execution.loss_scaling, self.config.model.dtype)

        dx, *_ = self.bwd_graph.bind(bwd_ts).call(
            loss_scaling, args=self.bwd_graph.grad_graph_info.inputs_dict(fwd_info)
        )

        return x, dx


class GPTLMHeadModelTP2D(addons.Module):
    def __init__(self, config: GPTConfig):
        """GPT model (GPT2-GPT3 architecture) with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = GPTModelTP2D(config, include_layer_norm=False)  # move layer norm to the head
        self.head = GPTLMHeadTP2D(config)

    def build(self, input_ids: popxl.Tensor, position_ids: popxl.Tensor = None) -> popxl.Tensor:
        x = self.transformer(input_ids, position_ids)
        x = self.head(x, self.transformer.embeddings.word.weight)
        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFGPT2LMHeadModel) -> WeightsDict:
        weights = GPTModelTP2D.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False)
        weights.update(GPTLMHeadTP2D.hf_mapping(config, variables.head, hf_model.transformer))
        return weights
