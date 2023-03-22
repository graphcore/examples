# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
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
"""
This script has been adapted from some of the original EspNet found here:
[
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/swish.py
]
"""
import torch


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swish activation function."""
        return x * torch.sigmoid(x)
