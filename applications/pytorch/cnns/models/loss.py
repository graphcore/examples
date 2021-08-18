# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from functools import reduce

import torch
from torch.nn.functional import nll_loss


def weighted_nll_loss(log_preds, labels, weights):
    """Compute a weighted nll loss based on multiple labels.

    Parameters:
        log_preds torch.Tensor: Logarithm of model predictions.
        labels List[torch.Tensor]: A list of label tensors, each tensor
            represents a distinct target.
        weights List[torch.Tensor]: a list of weight tensors, one for each
            target.
    """
    assert len(labels) == len(weights), (
        "'labels' and 'weights' must have the same number of tensors.")
    losses = [nll_loss(log_preds, label, reduction='none') for label in labels]
    final_loss = reduce(torch.add, [w * l for w, l in zip(weights, losses)])
    return torch.mean(final_loss)
