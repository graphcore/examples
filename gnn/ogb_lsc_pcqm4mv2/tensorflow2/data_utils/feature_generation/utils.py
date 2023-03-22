# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# This file has been modified by Graphcore Ltd.

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def normalize_edge_index(edge_index, edge_weight, deg, num_nodes):
    # L = D - A.
    loop_index = np.arange(num_nodes)
    loop_index = np.tile(loop_index, (2, 1))
    edge_index = np.concatenate((edge_index, loop_index), axis=1)
    edge_weight = np.concatenate((-edge_weight, deg), axis=0)

    return edge_index, edge_weight


def remove_self_loops(edge_index, weights):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if weights is None:
        return edge_index, None
    else:
        return edge_index, weights[mask]


def get_laplacian(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float16)

    row, col = edge_index[0], edge_index[1]

    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_nodes)

    edge_index, edge_weight = normalize_edge_index(edge_index, edge_weight, deg, num_nodes)
    return edge_index, edge_weight


def get_out_degrees(edges, num_nodes):
    return np.bincount(edges[0], minlength=num_nodes)


def get_in_degrees(edges, num_nodes):
    return np.bincount(edges[1], minlength=num_nodes)


def safe_inv(in_array):
    with np.errstate(divide="ignore"):
        out_array = 1 / in_array
    out_array[out_array == np.inf] = 0
    return out_array


def edges_to_dense_adjacency(edges, num_nodes):
    adj = sp.csr_matrix(
        (np.ones((edges.shape[1])), (edges[0, :], edges[1, :])), shape=(num_nodes, num_nodes), dtype=np.int32
    )
    return adj.toarray()
