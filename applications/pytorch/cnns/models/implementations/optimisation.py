# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch


class PaddedConv(torch.nn.Conv2d):
    """
    This layer can be applied as the first Conv2d layer.
    Expects 3 input channel and converts it for 4 channel for the Conv2d layer.
    The inputs forth channel is padded with zeros.
    """
    def __init__(self, layer):
        super().__init__(4, layer.out_channels, layer.kernel_size, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups, bias=hasattr(layer, "bias"), padding_mode=layer.padding_mode)
        self.extract_weights(layer)

    def forward(self, input):
        pad_shape = list(input.size())
        pad_shape[1] = 1
        padding = torch.zeros(pad_shape, dtype=input.dtype)
        padded_input = torch.cat((input, padding), 1)
        return super().forward(padded_input)

    def extract_weights(self, conv_layer):
        self.weight.data[:, :3, :, :] = conv_layer.weight.data
        if hasattr(conv_layer, "bias"):
            self.bias = conv_layer.bias
