# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2022 Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Dominique Beaini
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file has been modified by Graphcore Ltd.

import numpy as np

from data_utils.feature_generation.utils import edges_to_dense_adjacency, get_out_degrees, safe_inv


def get_random_walk_landing_probs(edges, num_nodes, k_steps=[1], edge_weights=None, space_dim=0):
    if edge_weights is None:
        edge_weights = np.ones(edges.shape[1])

    out_degrees = get_out_degrees(edges, num_nodes)
    out_degrees_inv = safe_inv(out_degrees)

    if len(edges) == 0:
        P = np.zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        # This is doing row-normalize of the adjacency matrix
        adj = edges_to_dense_adjacency(edges, num_nodes)
        P = np.diag(out_degrees_inv) @ adj

    random_walks = []
    for k in k_steps:
        res = np.linalg.matrix_power(P, k)
        res = np.diagonal(res, axis1=-2, axis2=-1) * (k ** (space_dim / 2))
        random_walks.append(res)

    random_walk_landing_probs = np.vstack(random_walks)
    random_walk_landing_probs = np.transpose(random_walk_landing_probs)

    assert random_walk_landing_probs.shape == (num_nodes, len(k_steps))

    return random_walk_landing_probs


def get_random_walk_landing_probs_from_dataset(dataset_item, item_options):
    random_walk_landing_probs = get_random_walk_landing_probs(
        dataset_item["edge_index"], dataset_item["num_nodes"], **item_options
    )
    return (random_walk_landing_probs,)
