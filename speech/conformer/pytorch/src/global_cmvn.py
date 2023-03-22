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
"""
This script has been adapted from some of the original EspNet found here:
[
    https://github.com/espnet/espnet/blob/master/espnet2/layers/global_mvn.py
]

Main changes:
    increase the part of generated data
    replce the processing of taking the reciprocal of std to pass the inv_std

"""


import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from src.utils.mask import make_pad_mask


class GlobalMVN(torch.nn.Module):
    """Apply global mean and variance normalization
    Args:
        stats_file: npy file
        norm_means: Apply mean normalization
        norm_vars: Apply var normalization
        eps:
    """

    def __init__(
        self,
        use_generate=False,
        feature_len=80,
        mean=None,
        inv_std=None,
    ):
        super().__init__()
        if use_generate:
            mean = np.random.randint(-20, -10, feature_len)
            std = np.random.randint(1, 6, feature_len)
            inv_std = torch.tensor(std)  # float(1/std))
            mean = torch.tensor(mean)
        assert mean.shape == inv_std.shape
        self.register_buffer("mean", mean)
        self.register_buffer("inv_std", inv_std)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        x = x - self.mean
        x = x * self.inv_std
        x = x.float()
        return x, ilens
