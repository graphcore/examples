# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    # provide labels only for samples, where prediction is available
    # (during the training, not every samples prediction is returned
    # for efficiency reasons by default)
    labels = labels[-predictions.size()[0] :]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0]
    return accuracy
