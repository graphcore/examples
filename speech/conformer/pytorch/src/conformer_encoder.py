# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2020 Tomoki Hayashi
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
#
# This file has been modified by Graphcore Ltd.

'''
This script has been adapted from some of the original EspNet found here:
[
    https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/conformer_encoder.py
]

Main changes:
    remove the subsample class: conv2d2, conv2d6, conv2d8
    remove intermediate_outs block

'''

import torch
from src.layers.attention import RelPositionMultiHeadedAttention
from src.layers.embedding import RelPositionalEncoding
from src.layers.swish import Swish
from src.layers.subsampling import Conv2dSubsampling
from src.layers.layer_norm import LayerNorm
from src.layers.convolution import ConvolutionModule
from src.layers.positionwise_feed_forward import PositionwiseFeedForward
from src.utils.repeat import repeat
from src.layers.encoder_layer import EncoderLayer
from src.utils.mask import make_pad_mask


class ConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        concat_after: bool = False,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        max_len: int = 512,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.dtype = dtype
        self.normalize_before = normalize_before
        activation = Swish()
        pos_enc_class = RelPositionalEncoding

        self.embed = Conv2dSubsampling(
                        input_size,
                        output_size,
                        dropout_rate,
                        pos_enc_class(output_size, positional_dropout_rate, max_len=max_len//4-1)
                    )

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )

        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            zero_triu,
        )

        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size, cnn_module_kernel, activation
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def forward(self, xs_pad, ilens):
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).

        """
        masks = (~make_pad_mask(ilens, xs_pad.size(1))[:, None, :])
        xs_pad, masks = self.embed(xs_pad, masks)
        
        # explicitly cast pos_emb to have the same type as batch
        if isinstance(xs_pad, tuple):
            xs_pad = (xs_pad[0], xs_pad[1].type(xs_pad[0].dtype))

        xs_pad, masks = self.encoders(xs_pad, masks)
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1, dtype=torch.int32)  # have to use int 32 for poptorch

        return xs_pad, olens, masks
