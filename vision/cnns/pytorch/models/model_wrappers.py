# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import poptorch
from functools import partial
from typing import List


class OverlapModel(torch.nn.Module):
    """Wraps the model to use IO tiles to overlap the IO with compute"""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, img):
        img = poptorch.set_overlap_for_input(img, poptorch.OverlapMode.OverlapAccumulationLoop)
        img = self.model(img)
        img = poptorch.set_overlap_for_output(img, poptorch.OverlapMode.OverlapAccumulationLoop)
        return img


class NormalizeInputModel(torch.nn.Module):
    """Wraps the model and convert the input tensor to the given type, and normalise it."""
    def __init__(self, model: torch.nn.Module, mean: List[float], std: List[float], output_cast=None):
        super().__init__()
        self.model = model
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.mul = (1.0/(255.0 * std)).view(-1, 1, 1)
        self.sub = (mean / std).view(-1, 1, 1)
        self.output_cast = output_cast
        if output_cast == "full":
            self.mul, self.sub = self.mul.float(), self.sub.float()
        elif output_cast == "half":
            self.mul, self.sub = self.mul.half(), self.sub.half()

    def forward(self, img):
        if self.output_cast == "half":
            img = img.half()
        elif self.output_cast == "full":
            img = img.float()
        img = img.mul(self.mul)
        img = img.sub(self.sub)
        return self.model(img)
