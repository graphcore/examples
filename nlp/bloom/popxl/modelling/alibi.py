# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
from typing import Any

import numpy as np


def build_alibi_data(max_seq_len: int, n_head: int, np_dtype: Any = "float32") -> np.ndarray:
    """
    alibi is a head-specific (shared by all blocks) constant bias applied to softmax scores
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    Args:
    Returns tensor shaped (n_head, 1, max_seq_len)
        max_seq_len: (`int`, *required*):
            max sequence length
        n_head: (`int`, *required*):
            number of heads
        dtype: (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))  # 2^(-8/n)
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = np.asarray(get_slopes(n_head), dtype=np_dtype)
    slopes = np.expand_dims(np.expand_dims(slopes, 1), 1)

    relative_dist = np.asarray(np.arange(max_seq_len), dtype=np_dtype)
    relative_dist = np.expand_dims(np.expand_dims(relative_dist, 0), 0)
    alibi_data = slopes * relative_dist
    return alibi_data
