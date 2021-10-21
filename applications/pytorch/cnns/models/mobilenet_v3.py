# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# @Author  : DevinYang(pistonyang@gmail.com)
# @Author  : QiangGong(qiangg@graphcore.ai)
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.
from functools import partial
from torch import nn
import math


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)


class hswish(nn.Module):
    def __init__(self):
        super(hswish, self).__init__()
        self.hsigmoid = hsigmoid()

    def forward(self, x):
        hsig = self.hsigmoid(x)
        out = x * hsig
        return out


class hsigmoid(nn.Module):
    def __init__(self):
        super(hsigmoid, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # replace relu6 with 2 relu as currently relu6 is not supported
        out = x + 3
        out = 6 - out
        out = self.relu(out)
        out = 6 - out
        out = self.relu(out)
        out = out / 6
        return out


class SE_Module(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_Module, self).__init__()
        reduction_c = make_divisible(channels // reduction)
        self.out = nn.Sequential(
            nn.Conv2d(channels, reduction_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_c, channels, 1, bias=True),
            hsigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.pool(x)
        y = self.out(y)
        return x * y


class MobileNetBottleneck(nn.Module):
    def __init__(self, in_c, expansion, out_c, kernel_size, stride, se=False,
                 activation='relu', first_conv=True, skip=True, linear=True, norm_layer=nn.BatchNorm2d):
        super(MobileNetBottleneck, self).__init__()

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'h_swish':
            self.act = hswish()
        hidden_c = round(in_c * expansion)
        self.linear = linear
        self.skip = stride == 1 and in_c == out_c and skip

        seq = []
        if first_conv and in_c != hidden_c:
            seq.append(nn.Conv2d(in_c, hidden_c, 1, 1, bias=False))
            seq.append(norm_layer(hidden_c))
            seq.append(self.act)
        seq.append(nn.Conv2d(hidden_c, hidden_c, kernel_size, stride,
                             kernel_size // 2, groups=hidden_c, bias=False))
        seq.append(norm_layer(hidden_c))
        seq.append(self.act)
        if se:
            seq.append(SE_Module(hidden_c))
        seq.append(nn.Conv2d(hidden_c, out_c, 1, 1, bias=False))
        seq.append(norm_layer(out_c))

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        skip = x
        x = self.seq(x)
        if self.skip:
            x = skip + x
        if not self.linear:
            x = self.act(x)
        return x


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000, small_input=False, dropout_rate=0.2, norm_layer=nn.BatchNorm2d):
        super(MobileNetV3_Large, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2 if not small_input else 1, 1, bias=False),
            norm_layer(16),
            hswish(),
        )
        self.mb_block = nn.Sequential(
            MobileNetBottleneck(16, 1, 16, 3, 1, False, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(16, 4, 24, 3, 2, False, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(24, 3, 24, 3, 1, False, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(24, 3, 40, 5, 2, True, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(40, 3, 40, 5, 1, True, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(40, 3, 40, 5, 1, True, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(40, 6, 80, 3, 2, False, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(80, 2.5, 80, 3, 1, False, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(80, 2.3, 80, 3, 1, False, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(80, 2.3, 80, 3, 1, False, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(80, 6, 112, 3, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(112, 6, 112, 3, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(112, 6, 160, 5, 2, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(160, 6, 160, 5, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(160, 6, 160, 5, 1, True, 'h_swish', norm_layer=norm_layer),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            norm_layer(960),
            hswish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1, bias=False),
            hswish(),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.Flatten(),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, small_input=False, dropout_rate=0.2, norm_layer=nn.BatchNorm2d):
        super(MobileNetV3_Small, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2 if not small_input else 1, 1, bias=False),
            norm_layer(16),
            hswish(),
        )
        self.mb_block = nn.Sequential(
            MobileNetBottleneck(16, 1, 16, 3, 2, True, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(16, 4.5, 24, 3, 2, False, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(24, 88 / 24, 24, 3, 1, False, 'relu', norm_layer=norm_layer),
            MobileNetBottleneck(24, 4, 40, 5, 2, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(40, 6, 40, 5, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(40, 6, 40, 5, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(40, 3, 48, 5, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(48, 3, 48, 5, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(48, 6, 96, 5, 2, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(96, 6, 96, 5, 1, True, 'h_swish', norm_layer=norm_layer),
            MobileNetBottleneck(96, 6, 96, 5, 1, True, 'h_swish', norm_layer=norm_layer),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            norm_layer(576),
            hswish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 1280, 1, bias=False),
            hswish(),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.Flatten(),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x
