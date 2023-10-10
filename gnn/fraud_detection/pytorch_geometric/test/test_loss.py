# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import torch.nn.functional as F

from loss import weighted_cross_entropy


def test_loss():
    ins = torch.randn(10, 2, requires_grad=True)
    target = torch.empty(10, dtype=torch.long).random_(2)
    mask = torch.randint(0, 2, (10,))

    expected_loss = F.cross_entropy(ins, torch.where(mask.bool(), target, -100))

    loss = weighted_cross_entropy(ins, target, mask)

    assert torch.isclose(loss, expected_loss)


def test_loss_with_class_weight():
    ins = torch.randn(10, 2, requires_grad=True)
    target = torch.empty(10, dtype=torch.long).random_(2)
    mask = torch.randint(0, 2, (10,))
    class_weight = (0.8, 0.2)

    expected_loss = F.cross_entropy(ins, torch.where(mask.bool(), target, -100), weight=torch.tensor(class_weight))

    loss = weighted_cross_entropy(ins, target, mask, weight=class_weight)

    assert torch.isclose(loss, expected_loss)


def test_loss_with_class_weight_and_batch_size():

    batch_size = 4

    ins = torch.randn(10, 2, requires_grad=True)
    target = torch.empty(10, dtype=torch.long).random_(2)
    mask = (target * 0).bool()
    mask[:batch_size] = 1
    class_weight = (0.8, 0.2)

    expected_loss = F.cross_entropy(ins[:batch_size], target[:batch_size], weight=torch.tensor(class_weight))

    loss = weighted_cross_entropy(ins, target, mask, weight=class_weight)

    assert torch.isclose(loss, expected_loss)
