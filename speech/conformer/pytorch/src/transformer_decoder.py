# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2019 Shigeki Karita
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
This script has been adapted from some of the original EspNet repo found here:
[
    https://github.com/espnet/espnet/blob/master/espnet2/asr/decoder/transformer_decoder.py
]
Main changes:
    Main change is the part of parameters of TransformerDecoder class.
'''

import torch
from src.layers.layer_norm import LayerNorm
from src.layers.positionwise_feed_forward import PositionwiseFeedForward
from src.utils.repeat import repeat
from src.layers.decoder_layer import DecoderLayer
from src.layers.embedding import PositionalEncoding
from src.layers.attention import MultiHeadedAttention
from src.utils.mask import make_pad_mask
from src.utils.mask import subsequent_mask
from typing import List, Optional, Tuple


class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        use_output_layer: bool = True,
        normalize_before: bool = True,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        concat_after: bool = False,
        max_len: int = 50,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.dtype = dtype
        attention_dim = encoder_output_size
        pos_enc_class = PositionalEncoding
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, attention_dim),
            pos_enc_class(attention_dim, positional_dropout_rate, max_len=max_len)  # add max_len to params
        )
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size, bias=True)
        else:
            self.output_layer = None
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.use_output_layer = use_output_layer

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens):
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt_mask = (~make_pad_mask(ys_in_lens, ys_in_pad.size(1))[:, None, :])
        m = subsequent_mask(tgt_mask.size(-1), tgt_mask.device).unsqueeze(0)
        tgt_attention_mask = (tgt_mask.int().repeat(1, m.size(1), 1) + m.int()).eq(2)
        memory_mask = (~make_pad_mask(hlens, hs_pad.size(1)))[:, None, :]
        x = self.embed(ys_in_pad)

        if self.dtype == torch.float16:
            x = x.type(torch.float16)

        x, tgt_attention_mask, hs_pad, memory_mask = self.decoders(
            x, tgt_attention_mask, hs_pad, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        # olens = tgt_mask.int().sum(1, dtype=torch.int32) ## this caused the problem 80 to 30 , it's too much wired
        return x, tgt_mask.squeeze(1)

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache
