# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Dict
import numpy as np
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertPooler

import popxl
from popxl import ops

import popxl_addons as addons
from config import BertConfig
from popxl_addons import NamedTensors
from .attention import SelfAttention
from .feed_forward import FeedForward
from .mlm import BertMLM
from .nsp import BertNSP


class BertLayer(addons.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.attention = SelfAttention(self.config)
        self.feed_forward = FeedForward(self.config)

    def build(self, x: popxl.Tensor, mask: popxl.Tensor, seed: Optional[popxl.Tensor] = None):
        attention_seed = None
        if seed is not None:
            seed, attention_seed = ops.split_random_seed(seed)

        x = self.attention(x, mask, attention_seed)
        x = self.feed_forward(x, seed)
        return x


class BertPretrainingLossAndGrad(addons.Module):
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.fwd = addons.Module()
        self.grad = addons.Module()

    def build(
        self,
        x: popxl.Tensor,
        word_embedding_t: popxl.Tensor,
        word_embedding_accum_t: popxl.TensorByRef,
        masked_positions: popxl.Tensor,
        mlm_labels: popxl.Tensor,
        nsp_labels: popxl.Tensor,
    ) -> Tuple[popxl.Tensor, popxl.Tensor]:

        mlm_loss, mlm_dx = self.mlm(x, word_embedding_t, word_embedding_accum_t, masked_positions, mlm_labels)
        nsp_loss, nsp_dx = self.nsp(x, nsp_labels)

        loss = mlm_loss + nsp_loss
        dx = mlm_dx + nsp_dx

        return loss, dx

    def mlm(
        self,
        x: popxl.Tensor,
        word_embedding_t: popxl.Tensor,
        word_embedding_accum_t: popxl.TensorByRef,
        masked_positions: popxl.Tensor,
        labels: popxl.Tensor,
    ):
        args, graph = BertMLM(self.config).create_graph(
            x, word_embedding_t=word_embedding_t, masked_positions=masked_positions
        )
        accums = list(graph.args.tensors) + [graph.graph.inputs[1]]  # layer norm weights + tied weight
        dargs, dgraph = addons.transforms.autodiff_with_accumulation(
            graph, accums, grads_required=[graph.graph.inputs[0]]
        )

        fwd_info = graph.bind(self.fwd.add_variable_inputs("mlm", args)).call_with_info(
            x, word_embedding_t, masked_positions
        )
        x = fwd_info.parent_output(0)

        loss, dx = addons.cross_entropy_with_grad(
            x, labels, loss_scaling=self.config.execution.loss_scaling, ignore_index=0
        )

        input_dict = dgraph.grad_graph_info.inputs_dict(fwd_info)
        input_dict.update({dgraph.args.accum.word_embedding_t: word_embedding_accum_t})

        (dx,) = dgraph.bind(self.grad.add_variable_inputs("mlm", dargs)).call(dx, args=input_dict)

        return loss, dx

    def nsp(self, x: popxl.Tensor, labels: popxl.Tensor):
        args, graph = BertNSP(self.config).create_graph(x)
        dargs, dgraph = addons.transforms.autodiff_with_accumulation(
            graph, graph.args.tensors, grads_required=[graph.graph.inputs[0]]
        )

        fwd_info = graph.bind(self.fwd.add_variable_inputs("nsp", args)).call_with_info(x)
        x = fwd_info.parent_output(0)

        loss, dx = addons.cross_entropy_with_grad(x, labels, loss_scaling=self.config.execution.loss_scaling)

        (dx,) = dgraph.bind(self.grad.add_variable_inputs("nsp", dargs)).call(
            dx, args=dgraph.grad_graph_info.inputs_dict(fwd_info)
        )

        return loss, dx

    @staticmethod
    def hf_mapping(
        config: BertConfig, variables: NamedTensors, hf_model: BertPreTrainingHeads, hf_model_pooler: BertPooler
    ) -> Dict[popxl.Tensor, np.ndarray]:
        return {
            **BertMLM.hf_mapping(config, variables.fwd.mlm, hf_model),
            **BertNSP.hf_mapping(config, variables.fwd.nsp, hf_model, hf_model_pooler),
        }
