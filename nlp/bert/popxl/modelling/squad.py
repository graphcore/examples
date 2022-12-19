# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Tuple, Dict
import numpy as np

from torch.nn.modules.linear import Linear as PTLinear

import popxl
from popxl import ops
from popxl.utils import to_numpy

import popxl_addons as addons
from config import BertConfig
from popxl_addons import NamedTensors
from popxl_addons.layers import Linear


class BertSquadHead(addons.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.qa_outputs = Linear(2)

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        logits = self.qa_outputs(x)
        # To obtain: `start_logits, end_logits = ops.split(logits, 2, -1)`
        return logits

    @staticmethod
    def hf_mapping(config: BertConfig, variables: NamedTensors, hf_model: PTLinear) -> Dict[popxl.Tensor, np.ndarray]:
        dtype = config.model.dtype

        return {
            variables.qa_outputs.weight: np.ascontiguousarray(to_numpy(hf_model.weight.data.T, dtype)),
            variables.qa_outputs.bias: np.ascontiguousarray(to_numpy(hf_model.bias.data, dtype)),
        }


class BertSquadLossAndGrad(addons.Module):
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def build(self, x: popxl.Tensor, labels: popxl.Tensor) -> Tuple[popxl.Tensor, popxl.Tensor]:
        args, graph = BertSquadHead(self.config).create_graph(x)
        dargs, dgraph = addons.transforms.autodiff_with_accumulation(
            graph, graph.args.tensors, grads_required=[graph.graph.inputs[0]])

        fwd_info = graph.bind(self.add_variable_inputs("fwd", args)).call_with_info(x)
        x = fwd_info.parent_output(0)

        loss, dx = self.loss(x, labels)

        dx, = dgraph.bind(self.add_variable_inputs("grad", dargs)).call(
            dx,
            args=dgraph.grad_graph_info.inputs_dict(fwd_info))

        return loss, dx

    def loss(self, x: popxl.Tensor, labels: popxl.Tensor):
        x = x.reshape_((-1, self.config.model.sequence_length, 2))
        starts, ends = ops.split(x, 2, axis=-1)
        start_labels, end_labels = ops.split(labels, 2, axis=-1)

        start_loss, start_dx = addons.cross_entropy_with_grad(
            ops.squeeze(starts, [-1]),
            ops.squeeze(start_labels, [-1]),
            loss_scaling=self.config.execution.loss_scaling / 2,
            ignore_index=self.config.model.sequence_length)

        end_loss, end_dx = addons.cross_entropy_with_grad(
            ops.squeeze(ends, [-1]),
            ops.squeeze(end_labels, [-1]),
            loss_scaling=self.config.execution.loss_scaling / 2,
            ignore_index=self.config.model.sequence_length)

        loss = (start_loss + end_loss) / 2
        dx = ops.concat_(
            (start_dx.reshape((-1, 1)), end_dx.reshape_((-1, 1))),
            axis=-1)

        return loss, dx
