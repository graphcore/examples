# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Tuple, Dict
import numpy as np

import popxl
from popxl import ops
from popxl.utils import to_numpy
from modelling import gpt_model

import popxl_addons as addons
from popxl_addons import WeightsDict
from config import GPTConfig
from popxl_addons import NamedTensors
from popxl_addons.layers import Linear, LayerNorm
from utils.utils import replica_groups


class GPTMnliHead(addons.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Replicated
        self.ln_f = LayerNorm()
        self.score = Linear(config.inference.mnli_n_classes)

    def build(self, x: popxl.Tensor, unpadded_length: popxl.Tensor) -> popxl.Tensor:
        # x: input is replicated
        x = self.ln_f(x)
        # Ensure flattened in batch by sequence dims
        x = x.reshape(
            (self.config.execution.micro_batch_size * self.config.model.sequence_length, self.config.model.hidden_size)
        )
        # Take hidden state from last token in sequence for each micro-batch and so ignore padding
        # popxl doesn't support multi-index integer indexing - so use a flattened index
        idx = (
            unpadded_length - 1 + self.config.model.sequence_length * np.arange(self.config.execution.micro_batch_size)
        )
        x = ops.gather(x, idx, axis=0)
        logits = self.score(x)
        return logits

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: gpt_model, include_score=True) -> WeightsDict:
        dtype = config.model.dtype
        weights = WeightsDict(
            {
                variables.head.head.ln_f.weight: to_numpy(hf_model.transformer.ln_f.weight.data, dtype),
                variables.head.head.ln_f.bias: to_numpy(hf_model.transformer.ln_f.bias.data, dtype),
            }
        )
        if include_score:
            weights.update(
                {
                    variables.score.weight: to_numpy(hf_model.score.weight.data, dtype),
                    variables.score.bias: to_numpy(hf_model.score.bias.data, dtype),
                }
            )

        return weights


class ClassificationLoss(addons.Module):
    def __init__(self, config: GPTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def build(self, logits: popxl.Tensor, labels: popxl.Tensor):
        loss, dx = addons.cross_entropy_with_grad(
            logits,
            labels,
            loss_scaling=self.config.execution.loss_scaling,
        )
        self.aux["dx"] = dx
        return loss

    def build_grad(self, dloss: popxl.Tensor) -> popxl.Tensor:
        return self.aux["dx"]


class GPTMnliLossHead(addons.Module):
    def __init__(self, config: GPTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.head = GPTMnliHead(config)

    def build(self, x: popxl.Tensor, unpadded_length: popxl.Tensor, labels: popxl.Tensor) -> popxl.Tensor:
        logits = self.head(x, unpadded_length)
        (loss,) = self.call_module(
            ClassificationLoss(self.config), grads_required=lambda graph: graph.graph.inputs[:1]
        )(logits, labels)
        loss = ops.rename(loss, "loss_output")
        return loss, logits

    @staticmethod
    def hf_mapping(config: GPTConfig, variables: NamedTensors, hf_model: gpt_model, include_score=True) -> WeightsDict:
        return GPTMnliHead.hf_mapping(config, variables, hf_model, include_score)
