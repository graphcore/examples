# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import reduce
import torch
import poptorch
from torch.nn.functional import nll_loss
from typing import Callable, List, Any


def weighted_nll_loss(log_preds: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    """Compute a weighted nll loss based on multiple labels.

    Parameters:
        log_preds torch.Tensor: Logarithm of model predictions.
        labels List[torch.Tensor]: A list of label tensors, each tensor
            represents a distinct target.
        weights List[torch.Tensor]: a list of weight tensors, one for each
            target.
    """
    assert len(labels) == len(weights), "'labels' and 'weights' must have the same number of tensors."
    losses = [nll_loss(log_preds, label, reduction="none") for label in labels]
    final_loss = reduce(torch.add, [w * l for w, l in zip(weights, losses)])
    return torch.mean(final_loss)


class LabelSmoothing:
    def __init__(self, loss: Callable = torch.nn.NLLLoss(reduction="mean"), label_smoothing: float = 0.0):
        """Provide adjusted classification and smoothing loss"""
        self.label_smoothing = label_smoothing
        self.class_loss = loss

    def label_smoothing_loss(self, log_preds: torch.Tensor, _: Any):
        if not isinstance(log_preds, torch.Tensor):
            log_preds = log_preds[0]
        return -torch.mean(log_preds) * self.label_smoothing

    def classification_loss(self, output: torch.Tensor, labels: torch.Tensor):
        return self.class_loss(output, labels) * (1.0 - self.label_smoothing)

    def get_losses_list(self) -> List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        losses = [self.classification_loss]
        if self.label_smoothing > 0.0:
            losses.append(self.label_smoothing_loss)
        return losses


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        losses: List[Callable],
        metrics: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = [],
    ):
        """Provides the wrapper around the model, which adds the loss function to the model.
        Parameters:
            model: the wrapped inference model
            losses: list of loss functions, which has input of (log(pred), coeffs) labels
            metrics: list of metric functions, which has input of (output, labels)
        """
        super().__init__()
        self.model = model
        self.losses = losses
        self.metrics = metrics

    def forward(self, input_data, labels=None):
        output = self.model(input_data)
        if isinstance(output, tuple):
            pred, coeffs = output
        else:
            pred = output
        if labels is None:
            return pred
        log_preds = torch.nn.functional.log_softmax(pred.float(), dim=1)
        if isinstance(output, tuple):
            loss_input = log_preds, coeffs
        else:
            loss_input = log_preds
        losses = [loss_fn(loss_input, labels) for loss_fn in self.losses]
        with torch.no_grad():
            metric_values = [metric_fn(pred, labels) for metric_fn in self.metrics]

        return poptorch.identity_loss(sum(losses), reduction="none"), tuple(losses), tuple(metric_values)
