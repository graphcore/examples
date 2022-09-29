# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from functools import partial
import numpy as np
from typing import Dict, Tuple, Callable, List, Union

# HF
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as HFModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons import NamedTensors, GraphWithNamedArgs
from popxl_addons.layers import Embedding
from popxl_addons.ops.replicated_all_reduce_TP import replicated_all_reduce_identical_inputs
from popxl_addons.ops.cross_entropy_sharded_loss import cross_entropy_sharded_loss
from popxl_addons.layers import LayerNorm
from popxl_addons.named_replica_grouping import fill_none_group

from modelling.gpt_model import GPTModelTP
from config import GPTConfig


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
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)
        # identical
        self.ln_f = LayerNorm()

    def build(self, x: popxl.Tensor, word_embedding: popxl.Tensor) -> popxl.Tensor:
        x = self.ln_f(x)
        # sharded
        x = replicated_all_reduce_identical_inputs(x, group=self.replica_grouping.transpose())
        logits = x @ word_embedding
        return logits

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: GPT2Model) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = {
            variables.ln_f.weight: to_numpy(hf_model.ln_f.weight.data, dtype),
            variables.ln_f.bias: to_numpy(hf_model.ln_f.bias.data, dtype),
        }

        return weights


class GPTLMHeadModelTP(addons.Module):
    def __init__(self, config: GPTConfig):
        """GPT model (GPT2-GPT3 architecture) with a language model head, with sharded weights."""
        super().__init__()
        self.config = config

        self.transformer = GPTModelTP(config, include_layer_norm=False)  # move layer norm to the head
        self.lm_head = GPTLMHeadTP(config)

    def build(self, input_ids: popxl.Tensor, position_ids: popxl.Tensor = None) -> popxl.Tensor:

        x = self.transformer(input_ids, position_ids)
        word_embedding = self.transformer.embeddings.word.weight.T
        x = self.lm_head(x, word_embedding)

        return x

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: HFModel) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype
        weights = GPTModelTP.hf_mapping(config, variables.transformer, hf_model.transformer, layer_norm=False)
        weights.update(GPTLMHeadTP.hf_mapping(config, variables.lm_head, hf_model.transformer))

        return weights


class GPTLMHeadLossAndGradTP(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        tp = config.execution.tensor_parallel
        dp = config.execution.data_parallel
        self.replica_grouping = popxl.gcg().ir.replica_grouping(stride=tp, group_size=dp)

    def build(self, x: popxl.Tensor, labels: popxl.Tensor, word_embedding_t: popxl.Tensor,
              word_embedding_accum_t: popxl.TensorByRef, word_offset: popxl.Tensor):

        vocab_shard_size = Embedding.get_vocab_shard_size(self.config.model.embedding.vocab_size,
                                                          self.config.execution.tensor_parallel)
        word_embed = popxl.TensorSpec((self.config.model.hidden_size, vocab_shard_size), dtype=x.dtype)

        fwd_facts, fwd_graph = GPTLMHeadTP(self.config).create_graph(x, word_embed)
        ts = self.add_variable_inputs("fwd", fwd_facts)

        ignore_index = -1 * word_offset
        # TODO: make cross_entropy_sharded_loss_with_grad and use float32 for loss_scaling, for consistency with popart
        loss_graph = GraphWithNamedArgs(
            fwd_graph.graph._ir.create_graph(cross_entropy_sharded_loss,
                                             fwd_graph.graph.outputs[0],
                                             labels,
                                             ignore_index=ignore_index,
                                             reduction='mean',
                                             replica_grouping=self.replica_grouping.transpose()))

        required_grads = [fwd_graph.graph.inputs[0]]
        accums = list(fwd_graph.args.tensors) + [fwd_graph.graph.inputs[1]]  # layer norm weights + tied weight
        none_is_all_replica = partial(fill_none_group, none_value=popxl.gcg().ir.replica_grouping())
        replica_groupings = fwd_facts.replica_groupings.map(none_is_all_replica)
        replica_groupings.insert('word_embedding', self.replica_grouping)

        bwd_facts, bwd_graph = addons.transforms.autodiff_with_accumulation(fwd_graph,
                                                                            tensors_to_accumulate_grads=accums,
                                                                            grads_required=required_grads,
                                                                            replica_groupings=replica_groupings)
        loss_bwd = addons.transforms.autodiff(loss_graph, grads_required=(loss_graph.graph.inputs[0], ))

        tied_weight = word_embedding_t
        fwd_info = fwd_graph.bind(ts).call_with_info(x, tied_weight)

        logits = fwd_info.parent_output(0)
        loss_fwd_info = loss_graph.call_with_info(logits, labels, ignore_index)
        loss = loss_fwd_info.parent_output(0)

        loss_scaling = popxl.constant(self.config.execution.loss_scaling, self.config.model.dtype)

        dx, = loss_bwd.call(loss_scaling, args=loss_bwd.grad_graph_info.inputs_dict(loss_fwd_info))

        ln_facts = bwd_facts.copy()
        ln_facts.accum.pop('word_embedding')
        bwd_weights = self.add_variable_inputs("bwd", ln_facts)

        input_dict = bwd_graph.grad_graph_info.inputs_dict(fwd_info)
        input_dict.update({bwd_graph.args.accum.word_embedding: word_embedding_accum_t})

        dx, = bwd_graph.bind(bwd_weights).call(dx, args=input_dict)

        return loss, dx
