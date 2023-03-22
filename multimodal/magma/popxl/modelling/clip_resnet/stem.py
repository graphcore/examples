# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np

from popxl_addons import Module, NamedTensors
from popxl_addons.layers import Conv2D
import popxl
from popxl import Tensor
from popxl import ops
from popxl.utils import to_numpy
from clip.model import ModifiedResNet as ClipModifiedResNet
from configs import ResNetConfig
from .batch_norm import BatchNorm2D

__all__ = ["Stem"]


class Stem(Module):
    def __init__(self, config: ResNetConfig):
        """
        Stem block of CLIP ModifiedResNet.
        Inference only: batch normalisation layers can work with a baked running mean and running vars,
        but these values won't be updated.
        """
        super().__init__()
        self.config = config
        self.conv1 = Conv2D(self.config.width // 2, kernel_size=3, strides=(2, 2), paddings=(1, 1, 1, 1), bias=False)
        self.bn1 = BatchNorm2D()
        self.conv2 = Conv2D(self.config.width // 2, kernel_size=3, paddings=(1, 1, 1, 1), bias=False)
        self.bn2 = BatchNorm2D()
        self.conv3 = Conv2D(self.config.width, kernel_size=3, paddings=(1, 1, 1, 1), bias=False)
        self.bn3 = BatchNorm2D()

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(ops.relu(x))
        x = self.bn2(x)
        x = self.conv3(ops.relu(x))
        x = self.bn3(x)
        # NOTE: average pool in pytorch has stride default value = kernel size.
        # this is different in popxl so we need to set all parameters
        x = ops.average_pool(ops.relu(x), kernel_size=(2, 2), stride=(2, 2))
        return x

    @staticmethod
    def clip_mapping(clip_model: ClipModifiedResNet, variables: NamedTensors):
        state_dict = {
            variables.conv1.weight: to_numpy(clip_model.conv1.weight.data),
            variables.bn1.weight: to_numpy(clip_model.bn1.weight.data),
            variables.bn1.bias: to_numpy(clip_model.bn1.bias.data),
            variables.bn1.running_mean: to_numpy(clip_model.bn1.running_mean.data),
            variables.bn1.running_var: to_numpy(clip_model.bn1.running_var.data),
            variables.conv2.weight: to_numpy(clip_model.conv2.weight.data),
            variables.bn2.weight: to_numpy(clip_model.bn2.weight.data),
            variables.bn2.bias: to_numpy(clip_model.bn2.bias.data),
            variables.bn2.running_mean: to_numpy(clip_model.bn2.running_mean.data),
            variables.bn2.running_var: to_numpy(clip_model.bn2.running_var.data),
            variables.conv3.weight: to_numpy(clip_model.conv3.weight.data),
            variables.bn3.weight: to_numpy(clip_model.bn3.weight.data),
            variables.bn3.bias: to_numpy(clip_model.bn3.bias.data),
            variables.bn3.running_mean: to_numpy(clip_model.bn3.running_mean.data),
            variables.bn3.running_var: to_numpy(clip_model.bn3.running_var.data),
        }
        return state_dict
