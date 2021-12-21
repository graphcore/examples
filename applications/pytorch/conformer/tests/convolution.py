# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch import nn
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from src.layers.layer_norm import LayerNorm


class ConvolutionModule_cpu(ConvolutionModule):
    """ConvolutionModule in Conformer model.

    Args:
        channels_ (int): The number of channels_ of conv layers.
        kernel_size_ (int): Kernerl size of conv layers.

    """

    def __init__(self, channels_, kernel_size_, activation_=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule_cpu, self).__init__(channels=channels_, kernel_size=kernel_size_, activation=activation_)
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size_ - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels_,
            2 * channels_,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            1 * channels_,
            1 * channels_,
            kernel_size_,
            stride=1,
            padding=(kernel_size_ - 1) // 2,
            groups=channels_,
            bias=bias,
        )
        # Replace the original batch_norm with layer_norm
        self.norm = LayerNorm(1 * channels_, -2)
        self.pointwise_conv2 = nn.Conv1d(
            1 * channels_,
            channels_,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation_
