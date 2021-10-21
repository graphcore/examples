# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import silu as Swish

import poptorch

from ..layers import ResidualBlock, ConvNormAct, get_norm


class CSPNeck(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, num_reps: int, activation: nn.Module, norm: str, num_groups: int = None):
        super().__init__()

        self.bottleneck_conv1 = ConvNormAct(ch_in, ch_out, 1, 1, activation, norm, num_groups)
        self.conv1 = nn.Conv2d(ch_out, ch_out, 1, 1, bias=False)
        self.conv2 = ConvNormAct(2 * ch_out, ch_out, 1, 1, activation, norm, num_groups)

        self.norm = get_norm(norm, ch_out*2, num_groups)
        self.res_modules = nn.Sequential(
            *[ResidualBlock(ch_out, ch_out, activation, norm, num_groups, shortcut=False) for _ in range(num_reps)]
        )
        self.act = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck_conv1(x)
        y1 = self.res_modules(x)
        y2 = self.conv1(x)
        y = torch.cat((y1, y2), dim=1)
        y = self.act(self.norm(y))
        return self.conv2(y)


class CSPDown(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, num_reps: int, activation: nn.Module, norm: str, num_groups: int = None) -> None:
        super().__init__()

        hidden = int(ch_out/2)
        self.bottleneck_conv = ConvNormAct(ch_in, hidden, 3, 2, activation, norm, num_groups)
        self.csp_neck = CSPNeck(hidden*2, hidden, 3, activation, norm, num_groups)
        self.conv = ConvNormAct(hidden, ch_out, 3, 1, activation, norm, num_groups)

    def forward(self, pan_connection_input: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.bottleneck_conv(x)
        x = torch.cat((x, pan_connection_input), 1)
        x = self.csp_neck(x)
        p = self.conv(x)
        return p, x


class CSPUp(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, num_reps: int, activation: nn.Module, norm: str, num_groups: int = None, upsample: bool = True) -> None:
        super().__init__()

        self.conv1 = ConvNormAct(ch_in, ch_out, 1, 1, activation, norm, num_groups)
        self.bneck_csp = CSPNeck(ch_out*2, ch_out, num_reps, activation, norm, num_groups)
        if upsample:
            self.conv2 = ConvNormAct(ch_out, int(ch_out/2), 1, 1, activation, norm, num_groups)
            self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        else:
            self.conv2 = ConvNormAct(ch_out, ch_out*2, 3, 1, activation, norm, num_groups)
            self.upsample = None

    def forward(self, pan_connection_input: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.conv1(pan_connection_input)
        x = torch.cat((p, x), 1)
        p = self.bneck_csp(x)
        x = self.conv2(p)
        if self.upsample is not None:
            x = self.upsample(x)
        return p, x


class CSPSPP(nn.Module):
    """Spatial Pyramid Pooling using cross stage partial networks - https://arxiv.org/pdf/1903.08589.pdf"""
    def __init__(self, ch_in: int, ch_out: int, num_reps: int, activation: nn.Module, norm: str, num_groups: int = None, maxp_kernel_size: Tuple[int, ...] = (5, 9, 13)):
        super().__init__()

        self.conv_up2 = ConvNormAct(ch_in, ch_out, 1, 1, activation, norm, num_groups)

        self.conv_up1 = nn.Conv2d(ch_in, ch_out, 1, 1, bias=False)
        self.conv1 = ConvNormAct(ch_out, ch_out, 3, 1, activation, norm, num_groups)
        self.conv2 = ConvNormAct(ch_out, ch_out, 1, 1, activation, norm, num_groups)
        self.maxp_modules = nn.ModuleList([nn.MaxPool2d(kernel_size=kernel, stride=1, padding=(kernel // 2)) for kernel in maxp_kernel_size])
        self.maxp_bottleneck_conv = ConvNormAct(4 * ch_out, ch_out, 1, 1, activation, norm, num_groups)
        self.conv3 = ConvNormAct(ch_out, ch_out, 3, 1, activation, norm, num_groups)

        self.act = activation
        self.norm = get_norm(norm, ch_out*2, num_groups)
        self.cat_bottleneck = ConvNormAct(2 * ch_out, ch_out, 1, 1, activation, norm, num_groups)

        self.convup = ConvNormAct(ch_out, int(ch_out/2), 1, 1, activation, norm, num_groups)
        self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y2 = self.conv_up1(x)

        x = self.conv_up2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        y1 = torch.cat([x] + [maxp(x) for maxp in self.maxp_modules], 1)
        y1 = self.maxp_bottleneck_conv(y1)
        y1 = self.conv3(y1)

        y = torch.cat((y1, y2), dim=1)
        y = self.act(self.norm(y))
        p5 = self.cat_bottleneck(y)

        x = self.convup(p5)
        x = self.upsample(x)
        return p5, x


class Yolov4P5Neck(nn.Module):
    """Yolov4-P5 neck as described in https://arxiv.org/abs/2011.08036"""
    def __init__(self, activation: nn.Module, norm: str = "group", number_groups: int = 2, calculate_loss: bool = False) -> None:
        super().__init__()

        self.SPP = CSPSPP(1024, 512, 1, activation, norm, number_groups, maxp_kernel_size=(5, 9, 13))
        self.cspUp1 = CSPUp(512, 256, 3, activation, norm, number_groups)
        self.cspUp2 = CSPUp(256, 128, 3, activation, norm, number_groups, upsample=False)
        self.cspDown1 = CSPDown(128, 512, 3, activation, norm, number_groups)
        self.cspDown2 = CSPDown(256, 1024, 3, activation, norm, number_groups)

        if calculate_loss:
            self.loss = nn.L1Loss()
        self.calculate_loss = calculate_loss

    def dummy_loss(self, act: torch.Tensor) -> torch.Tensor:
        return self.loss(act, target=torch.zeros_like(act))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        p5, p4, p3 = x

        p5, x = self.SPP(p5)
        p4, x = self.cspUp1(p4, x)
        x, p3 = self.cspUp2(p3, x)

        p4, x = self.cspDown1(p4, x)
        p5, x = self.cspDown2(p5, x)

        if self.calculate_loss:
            loss = poptorch.identity_loss(self.dummy_loss(p5) + self.dummy_loss(p4) + self.dummy_loss(p3), 'sum')
            return p5, p4, p3, loss
        return (p5, p4, p3)
