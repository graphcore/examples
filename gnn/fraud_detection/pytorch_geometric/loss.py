# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import poptorch
import torch.nn.functional as F


def weighted_cross_entropy(out, target, mask=None, weight=None):
    loss = F.cross_entropy(out, target, reduction="none")

    if weight is not None:
        weight = target * weight[1] + (1 - target) * weight[0]
        weight *= mask
        weight *= mask.sum() / weight.sum()
        loss *= weight

    loss *= mask
    loss = loss.sum() / mask.sum()
    return poptorch.identity_loss(loss, reduction="none")
