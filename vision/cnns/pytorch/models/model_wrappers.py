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

    def forward(self, *img):
        def parse(tensor, tensor_type="input"):
            if isinstance(tensor, torch.Tensor):
                if tensor_type == "input":
                    tensor = poptorch.set_overlap_for_input(tensor, poptorch.OverlapMode.OverlapAccumulationLoop)
                else:
                    tensor = poptorch.set_overlap_for_output(tensor, poptorch.OverlapMode.OverlapAccumulationLoop)
                return tensor
            else:
                tensor = tuple([parse(t, tensor_type) for t in tensor])
                return tensor

        img = parse(img, "input")
        img = self.model(*img)
        img = parse(img, "output")
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
        img = img.mul(self.mul.to(img.device))
        img = img.sub(self.sub.to(img.device))
        return self.model(img)
