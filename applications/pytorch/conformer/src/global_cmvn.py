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
# Main change is modifying the part of parameters of GlobalMVN class.

import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from src.utils.nets_utils import make_pad_mask


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
        use_generate: False,
        feature_len: 80,
        stats_file: str = '',
        norm_means: bool = True,
        norm_vars: bool = True,
        eps: float = 1.0e-3,
    ):
        super().__init__()
        if not use_generate:
            self.norm_means = norm_means
            self.norm_vars = norm_vars
            self.eps = eps
            stats_file = Path(stats_file)

            self.stats_file = stats_file
            stats = np.load(stats_file)
            if isinstance(stats, np.ndarray):
                count = stats[0].flatten()[-1]
                mean = stats[0, :-1] / count
                var = stats[1, :-1] / count - mean * mean
            else:
                count = stats["count"]
                sum_v = stats["sum"]
                sum_square_v = stats["sum_square"]
                mean = sum_v / count
                var = sum_square_v / count - mean * mean
            std = np.sqrt(np.maximum(var, eps))
        else:
            mean = np.random.randint(-20, -10, feature_len)
            std = np.random.randint(1, 6, feature_len)

        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("std", torch.from_numpy(std).float())


    def extra_repr(self):
        return (
            f"stats_file={self.stats_file}, "
            f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"
        )

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        x = x - self.mean
        x = torch.div(x, self.std, rounding_mode='trunc')
        x = x.float()
        return x, ilens
