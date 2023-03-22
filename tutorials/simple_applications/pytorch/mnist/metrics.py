# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    labels = labels[-predictions.size()[0] :]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy
