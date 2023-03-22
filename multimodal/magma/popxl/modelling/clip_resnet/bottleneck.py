# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from clip.model import Bottleneck as ClipBottleneck

from popxl_addons import Module, NamedTensors
from popxl_addons.layers import Conv2D
from popxl import ops, Tensor
from popxl.utils import to_numpy
from .batch_norm import BatchNorm2D
import popxl

__all__ = ["Bottleneck"]


class Downsample(Module):
    """
    Downsample section of a Bottleneck layer.
    Inference only: batch normalisation layers can work with a baked running mean and running vars,
    but these values won't be updated.
    """

    def __init__(self, planes, stride=1, expansion=4, cache: bool = False):
        super().__init__(cache)
        self.stride = stride
        self.conv = Conv2D(planes * expansion, kernel_size=1, strides=(1, 1), bias=False)
        self.bn = BatchNorm2D()

    def build(self, x: Tensor):
        # NOTE: average pool in pytorch has stride default value = kernel size.
        # this is different in popxl so we need to set all parameters
        x = ops.average_pool(x, kernel_size=(self.stride, self.stride), stride=(self.stride, self.stride))
        x = self.conv(x)
        x = self.bn(x)
        return x

    @staticmethod
    def clip_mapping(clip_model: ClipBottleneck, variables: NamedTensors):
        return {
            variables.conv.weight: to_numpy(clip_model.downsample[1].weight),
            variables.bn.weight: to_numpy(clip_model.downsample[2].weight),
            variables.bn.bias: to_numpy(clip_model.downsample[2].bias),
            variables.bn.running_mean: to_numpy(clip_model.downsample[2].running_mean),
            variables.bn.running_var: to_numpy(clip_model.downsample[2].running_var),
        }


class Bottleneck(Module):
    """
    Bottleneck layer of Clip ModifiedResnet.
    Inference only: batch normalisation layers can work with a baked running mean and running vars,
    but these values won't be updated.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cache=False):
        super().__init__(cache)

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = Conv2D(planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2D()

        self.conv2 = Conv2D(planes, kernel_size=(3, 3), paddings=(1, 1, 1, 1), bias=False)
        self.bn2 = BatchNorm2D()

        self.conv3 = Conv2D(planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2D()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = Downsample(planes, self.stride, Bottleneck.expansion, cache)

    def build(self, x: Tensor):
        identity = x
        out = ops.relu(self.bn1(self.conv1(x)))
        out = ops.relu(self.bn2(self.conv2(out)))
        if self.stride > 1:
            # NOTE: average pool in pytorch has stride default value = kernel size.
            # this is different in popxl so we need to set all parameters
            out = ops.average_pool(out, kernel_size=(self.stride, self.stride), stride=(self.stride, self.stride))

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)

        with popxl.in_sequence(True):
            out += identity
            out = ops.relu_(out)
        return out

    @staticmethod
    def clip_mapping(clip_model: ClipBottleneck, variables: NamedTensors):
        state_dict = {
            variables.conv1.weight: to_numpy(clip_model.conv1.weight),
            variables.bn1.weight: to_numpy(clip_model.bn1.weight),
            variables.bn1.bias: to_numpy(clip_model.bn1.bias),
            variables.bn1.running_mean: to_numpy(clip_model.bn1.running_mean),
            variables.bn1.running_var: to_numpy(clip_model.bn1.running_var),
            variables.conv2.weight: to_numpy(clip_model.conv2.weight),
            variables.bn2.weight: to_numpy(clip_model.bn2.weight),
            variables.bn2.bias: to_numpy(clip_model.bn2.bias),
            variables.bn2.running_mean: to_numpy(clip_model.bn2.running_mean),
            variables.bn2.running_var: to_numpy(clip_model.bn2.running_var),
            variables.conv3.weight: to_numpy(clip_model.conv3.weight),
            variables.bn3.weight: to_numpy(clip_model.bn3.weight),
            variables.bn3.bias: to_numpy(clip_model.bn3.bias),
            variables.bn3.running_mean: to_numpy(clip_model.bn3.running_mean),
            variables.bn3.running_var: to_numpy(clip_model.bn3.running_var),
        }
        if clip_model.downsample:
            state_dict.update(Downsample.clip_mapping(clip_model, variables.downsample))
        return state_dict
