# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Max Bain
# This file has been modified by Graphcore

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, x):
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        x = logit_scale * x
        labels = torch.Tensor(np.arange(x.shape[0])).long().to(x.device)
        t_loss = self.loss(x, labels)
        i_loss = self.loss(x.t(), labels)
        loss = (t_loss + i_loss) / 2.0
        return loss


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        # Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)

        # Sum over positives
        idiag = self.diag(i_logsm)
        loss_i = idiag.sum() / idiag.shape[0]

        jdiag = self.diag(j_logsm)
        loss_j = jdiag.sum() / jdiag.shape[0]

        return -loss_i - loss_j

    def diag(self, x):
        eyes = torch.eye(x.shape[0])
        return torch.sum(x * eyes, dim=-1)
