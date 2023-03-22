# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Union, Dict, Tuple

import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis

from models.backbone.yolov4_p5 import CrossStagePartialBlock
from models.layers import ConvNormAct

__all__ = ["Detector"]


class Detector(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for detection models.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module, detector_head: nn.Module):
        self.backbone = backbone
        self.neck = neck
        self.detector_head = detector_head

        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def output_shape(self, input_shape: Tuple[int, ...]) -> Dict[str, Tuple[int, ...]]:
        pass

    def _split_bn_layer(self, bn: nn.BatchNorm2d, start: int, end: int) -> nn.BatchNorm2d:
        norm = nn.BatchNorm2d(end - start, eps=bn.eps, momentum=bn.momentum, track_running_stats=bn.track_running_stats)
        norm.weight = nn.Parameter(bn.weight[start:end].clone())
        norm.bias = nn.Parameter(bn.bias[start:end].clone())
        norm.running_mean = nn.Parameter(bn.running_mean[start:end].clone())
        norm.running_var = nn.Parameter(bn.running_var[start:end].clone())
        return norm

    def optimize_for_inference(self, fuse_all_layers=True) -> None:
        # fuse conv with norm only when using BatchNorm. When using GroupNorm it will be turned off
        for m in self.modules():
            if isinstance(m, ConvNormAct):
                if isinstance(m.norm, nn.BatchNorm2d):
                    m.conv.weight, m.conv.bias = nn.utils.fusion.fuse_conv_bn_weights(
                        m.conv.weight,
                        m.conv.bias,
                        m.norm.running_mean,
                        m.norm.running_var,
                        m.norm.eps,
                        m.norm.weight,
                        m.norm.bias,
                    )
                    m.norm = nn.Identity()
            if fuse_all_layers and isinstance(m, CrossStagePartialBlock):
                if isinstance(m.norm, nn.BatchNorm2d):
                    # split norm layer into two parts and fuse with previous conv separately
                    temp_norm = self._split_bn_layer(m.norm, 0, m.hidden)
                    m.conv1.weight, m.conv1.bias = nn.utils.fusion.fuse_conv_bn_weights(
                        m.conv1.weight,
                        m.conv1.bias,
                        temp_norm.running_mean,
                        temp_norm.running_var,
                        temp_norm.eps,
                        temp_norm.weight,
                        temp_norm.bias,
                    )

                    temp_norm = self._split_bn_layer(m.norm, m.hidden, m.hidden * 2)
                    m.bottleneck_conv2.weight, m.bottleneck_conv2.bias = nn.utils.fusion.fuse_conv_bn_weights(
                        m.bottleneck_conv2.weight,
                        m.bottleneck_conv2.bias,
                        temp_norm.running_mean,
                        temp_norm.running_var,
                        temp_norm.eps,
                        temp_norm.weight,
                        temp_norm.bias,
                    )
                    m.norm = nn.Identity()

    def flop_count_analysis(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> FlopCountAnalysis:
        return FlopCountAnalysis(self, inputs)
