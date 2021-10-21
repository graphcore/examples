# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Optional, Tuple
import torch.nn as nn
import torch
import poptorch

from ..layers import ResidualBlock, ConvNormAct, get_norm


class CrossStagePartialBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, number_of_repetitions: int = 1, activation: nn.Module = nn.ReLU(), norm: str = "group", num_groups: int = None) -> None:
        super().__init__()

        self.ch_in = ch_in
        self.hidden = int(ch_out * 0.5)
        self.bottleneck_conv1 = ConvNormAct(
            self.ch_in, self.hidden, 1, 1, activation, norm, num_groups)
        self.bottleneck_conv2 = nn.Conv2d(self.ch_in, self.hidden, 1, 1, bias=False)

        self.conv1 = nn.Conv2d(self.hidden, self.hidden, 1, 1, bias=False)
        self.conv2 = ConvNormAct(2 * self.hidden, ch_out, 1, 1, activation, norm, num_groups)

        self.norm = get_norm(norm, self.hidden*2, num_groups)

        self.res_modules = nn.Sequential(
            *[ResidualBlock(self.hidden, self.hidden, activation, norm, num_groups) for _ in range(number_of_repetitions)])

        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.bottleneck_conv1(x)
        x2 = self.res_modules(x1)
        y1 = self.conv1(x2)

        y2 = self.bottleneck_conv2(x)

        y = torch.cat((y1, y2), dim=1)
        y = self.act(self.norm(y))
        return self.conv2(y)


class CSPDark(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, num_reps: int, activation: nn.Module, norm: str, num_groups: int = None) -> None:
        super().__init__()

        self.downsample = ConvNormAct(ch_in, ch_out, 3, 2, activation, norm, num_groups)
        self.csp = CrossStagePartialBlock(
            ch_out, ch_out, num_reps, activation, norm, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.csp(self.downsample(x))


class Yolov4P5BackBone(nn.Module):
    """Yolov4-P5 backbone as described in https://arxiv.org/abs/2011.08036"""
    def __init__(self, input_channels: int, activation: nn.Module, norm: str = "group", num_groups: int = 2, calculate_loss: bool = False) -> None:
        super().__init__()

        self.conv1 = ConvNormAct(input_channels, 32, 3, 1, activation, norm, num_groups)
        self.cspdark1 = CSPDark(32, 64, 1, activation, norm, num_groups)
        self.cspdark2 = CSPDark(64, 128, 3, activation, norm, num_groups)
        self.cspdark3 = CSPDark(128, 256, 15, activation, norm, num_groups)
        self.cspdark4 = CSPDark(256, 512, 15, activation, norm, num_groups)
        self.cspdark5 = CSPDark(512, 1024, 7, activation, norm, num_groups)

        if calculate_loss:
            self.loss = nn.L1Loss()
        self.calculate_loss = calculate_loss

    def dummy_loss(self, act: torch.Tensor) -> torch.Tensor:
        return self.loss(act, target=torch.zeros_like(act))

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        x = self.conv1(x)
        x = self.cspdark1(x)
        x = self.cspdark2(x)
        p3 = self.cspdark3(x)
        p4 = self.cspdark4(p3)
        p5 = self.cspdark5(p4)

        if self.calculate_loss:
            loss = poptorch.identity_loss(self.dummy_loss(p5) + self.dummy_loss(p4) + self.dummy_loss(p3), 'sum')
            return p5, p4, p3, loss

        return p5, p4, p3
