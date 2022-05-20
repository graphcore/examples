# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2022 The Google Research Authors.
#
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

import numpy as np
import scipy.sparse as sp


def construct_adj(edges, num_data, directed_graph):
    adj = sp.csr_matrix(
        (
            np.ones((edges.shape[0]), dtype=np.float32),
            (edges[:, 0], edges[:, 1])
        ),
        shape=(num_data, num_data))

    if not directed_graph:
        adj += adj.transpose()

    return adj


def sample_mask(idx, size):
    """Create mask."""
    mask = np.zeros(size)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
