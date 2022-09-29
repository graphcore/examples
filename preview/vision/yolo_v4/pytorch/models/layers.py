# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Tuple

import torch.nn as nn
import torch


def get_norm(norm: str, num_features: int, num_groups: int = None):
    if norm == "batch":
        return nn.BatchNorm2d(num_features, eps=0.001, momentum=0.03)
    elif norm == "group":
        return nn.GroupNorm(num_groups, num_features, eps=0.001)
    else:
        raise ValueError('Norm must be either \"batch\" or \"group\"')


class ConvNormAct(nn.Module):
    def __init__(self,
                 ch_in: int, ch_out: int, kernel_size: int = 1,
                 stride: int = 1, activation: nn.Module = nn.Identity(),
                 norm: str = "group", num_groups: int = None) -> None:
        super().__init__()

        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size,
                              stride, kernel_size // 2, groups=1, bias=False)
        self.norm = get_norm(norm, ch_out*1, num_groups)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "group",
                 num_groups: int = None,
                 shortcut: bool = True) -> None:
        super().__init__()

        self.conv1 = ConvNormAct(ch_in, ch_out, 1, 1, activation, norm, num_groups)
        self.conv2 = ConvNormAct(ch_out, ch_out, 3, 1, activation, norm, num_groups)
        self.add = shortcut and ch_in == ch_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class Mish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        softplus = torch.log(1 + torch.exp(x))
        return x * torch.tanh(softplus)
