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

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import tensorflow as tf
import torch
from torch_geometric.utils import remove_self_loops

from data_utils.feature_generation.generic_features import _preprocess_item_send_rcv
from data_utils.feature_generation.laplacian_features import eigvec_normalizer, get_laplacian_features
from data_utils.feature_generation.random_walk_features import edges_to_dense_adjacency, get_random_walk_landing_probs
from data_utils.feature_generation.utils import (
    get_in_degrees,
    get_out_degrees,
    normalize_edge_index,
    remove_self_loops,
    safe_inv,
)
from data_utils.load_dataset import GeneratedGraphData, GeneratedOGBGraphData
from data_utils.preprocess_dataset import preprocess_items


def get_example_edges(num_nodes):
    edges = np.vstack((np.arange(1, num_nodes), np.arange(0, num_nodes - 1)))
    edges = np.hstack((edges, np.flipud(edges)))
    return edges


@pytest.mark.parametrize("num_nodes", [1, 10, 100])
@pytest.mark.parametrize("max_freqs", [1, 3, 10, 20])
def test_get_laplacian_features(num_nodes, max_freqs):
    data = dict(num_nodes=num_nodes, edge_index=get_example_edges(num_nodes))
    eig_val, eig_vec = get_laplacian_features(data, max_freqs=max_freqs)
    assert eig_vec.shape == (num_nodes, max_freqs)
    assert eig_val.shape == (num_nodes, max_freqs, 1)


def test_get_laplacian_features_no_edges():
    num_nodes = 2
    max_freqs = 3
    edge_index = np.array([[], []])
    data = dict(num_nodes=num_nodes, edge_index=edge_index)
    eig_val, eig_vec = get_laplacian_features(data, max_freqs=max_freqs)
    assert eig_vec.shape == (num_nodes, max_freqs)
    assert eig_val.shape == (num_nodes, max_freqs, 1)


def test_eigvec_normalizer():
    num_frequencies = 10
    eps = 1e-12
    evals = np.random.random(num_frequencies)
    evects = np.random.random((num_frequencies, num_frequencies))

    # Normalize and pad eigen vectors.
    evects_l1 = eigvec_normalizer(evects, evals, normalization="L1")
    t_denom = torch.from_numpy(evects).norm(p=1, dim=0, keepdim=True)
    t_denom = t_denom.clamp_min(eps).expand_as(torch.from_numpy(evects))
    t_eig_vecs = torch.from_numpy(evects) / t_denom
    assert np.allclose(evects_l1, t_eig_vecs)

    evects_l2 = eigvec_normalizer(evects, evals, normalization="L2")
    t_denom = torch.from_numpy(evects).norm(p=2, dim=0, keepdim=True)
    t_denom = t_denom.clamp_min(eps).expand_as(torch.from_numpy(evects))
    t_eig_vecs = torch.from_numpy(evects) / t_denom
    assert np.allclose(evects_l2, t_eig_vecs)

    evects_abs = eigvec_normalizer(evects, evals, normalization="abs-max")
    t_denom = torch.max(torch.from_numpy(evects).abs(), dim=0, keepdim=True).values
    t_denom = t_denom.clamp_min(eps).expand_as(torch.from_numpy(evects))
    t_eig_vecs = torch.from_numpy(evects) / t_denom
    assert np.allclose(evects_abs, t_eig_vecs)


def test_normalize_edge_index():
    num_nodes = 3
    num_edges = 10
    edge_index, edge_weight = get_edge_index_and_weights(num_nodes, num_edges)

    # Add a 'floating' node for test purposes
    pad = np.array([[3, 3]])
    num_nodes += 1

    edge_index = np.concatenate((edge_index, pad.T), axis=1)
    edge_weight = np.append(edge_weight, 1)
    row, col = edge_index[0], edge_index[1]

    max_node_id = max(row)
    deg = tf.math.unsorted_segment_sum(edge_weight, row, max_node_id + 1)
    # L = D - A.
    t_edge_weight = torch.cat([torch.from_numpy(-edge_weight), torch.from_numpy(deg.numpy())], dim=0)

    edge_index, edge_weight = normalize_edge_index(edge_index, edge_weight, deg, num_nodes)

    # Expected values with the self loops added
    expected_edges = np.array(
        [
            [0.0, 1.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        ]
    )
    assert np.allclose(expected_edges, edge_index)
    assert np.allclose(t_edge_weight, edge_weight)


def get_edge_index_and_weights(num_nodes, num_edges):
    edge_index = GeneratedOGBGraphData.get_random_edge_idx(num_nodes, num_edges)
    edge_weights = np.ones(edge_index.shape[1], dtype=np.float16)
    return edge_index, edge_weights


def test_remove_self_loops():
    num_nodes = 3
    num_edges = 10
    orig_edge_index, _ = get_edge_index_and_weights(num_nodes, num_edges)
    # Add self-connections
    self_connections = np.linspace((0, 0), (num_nodes - 1, num_nodes - 1), num_nodes).T.astype(np.int64)
    edge_index = np.concatenate((orig_edge_index, self_connections), axis=1)
    edge_weights = np.ones(edge_index.shape[1], dtype=np.float16)
    edge_index, edge_weights = remove_self_loops(edge_index, edge_weights)

    assert np.all(edge_index == orig_edge_index)


def test_get_out_degrees():
    num_nodes = 10
    edges = np.array([[0, 1, 3, 4, 5, 4, 3, 2], [1, 0, 1, 4, 6, 0, 0, 4]])
    expected_out_degrees = np.array([1, 1, 1, 2, 2, 1, 0, 0, 0, 0])

    out_degrees = get_out_degrees(edges, num_nodes)

    np.testing.assert_array_equal(out_degrees, expected_out_degrees)


def test_get_in_degrees():
    num_nodes = 10
    edges = np.array([[0, 1, 3, 4, 5, 4, 3, 2], [1, 0, 1, 4, 6, 0, 0, 4]])
    expected_in_degrees = np.array([3, 2, 0, 0, 2, 0, 1, 0, 0, 0])

    in_degrees = get_in_degrees(edges, num_nodes)
    np.testing.assert_array_equal(in_degrees, expected_in_degrees)


def test_safe_inv():
    in_array = np.array([0, 1, 4, 0, 2])
    expected_inv = np.array([0.0, 1.0, 0.25, 0.0, 0.5])

    inv = safe_inv(in_array)
    np.testing.assert_array_equal(inv, expected_inv)


def test_edges_to_dense_adjacency():
    num_nodes = 6
    edges = np.array([[0, 1, 3, 4, 4, 3, 2], [1, 0, 1, 4, 0, 0, 4]])
    expected_adj = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    adj = edges_to_dense_adjacency(edges, num_nodes)
    np.testing.assert_array_equal(adj, expected_adj)

    # Test one node
    num_nodes = 1
    edges = np.array([[0], [0]])
    expected_adj = np.array([[1]])

    adj = edges_to_dense_adjacency(edges, num_nodes)
    np.testing.assert_array_equal(adj, expected_adj)


@pytest.mark.parametrize("num_nodes", [10, 15, 20, 23])
def test_random_walk_landing_probs(num_nodes):
    edges = get_example_edges(num_nodes)

    k_steps = [1]
    random_walk_landing_probs = get_random_walk_landing_probs(edges, num_nodes, k_steps)
    assert np.all(random_walk_landing_probs == 0)
    assert random_walk_landing_probs.shape == (num_nodes, len(k_steps))

    k_steps = [4, 6, 12, 24]
    random_walk_landing_probs = get_random_walk_landing_probs(edges, num_nodes, k_steps, space_dim=1)
    assert np.all(random_walk_landing_probs > 0)
    assert random_walk_landing_probs.shape == (num_nodes, len(k_steps))
    np.testing.assert_allclose(random_walk_landing_probs[:, 0], random_walk_landing_probs[:, 1], rtol=0.3)
    np.testing.assert_allclose(random_walk_landing_probs[:, 0], random_walk_landing_probs[:, 2], rtol=0.3)


@pytest.mark.parametrize("num_nodes", [1, 2])
@pytest.mark.parametrize("k_steps", [[1], [4, 6, 12, 24]])
def test_random_walk_landing_probs_small(num_nodes, k_steps):
    edges = get_example_edges(num_nodes)
    random_walk_landing_probs = get_random_walk_landing_probs(edges, num_nodes, k_steps, space_dim=1)
    assert random_walk_landing_probs.shape == (num_nodes, len(k_steps))


def test_senders_receivers():
    edges = np.array([[0, 1, 1, 0, 3, 1, 3, 4], [1, 0, 0, 1, 1, 3, 4, 3]])
    edge_feat = np.zeros([8, 3])
    expected_senders = [0, 1, 3, 3, 1, 0, 1, 4]
    expected_receivers = [1, 0, 1, 4, 0, 1, 3, 3]
    edges, senders, receivers = _preprocess_item_send_rcv(edge_feat, edges, bidirectional=True)
    np.testing.assert_array_equal(senders, expected_senders)
    np.testing.assert_array_equal(receivers, expected_receivers)


def test_cache():

    item_keys = ("test_out_1", "test_out_2")

    mock_dataset = GeneratedGraphData(total_num_graphs=100, nodes_per_graph=3, edges_per_graph=6)

    def preprocess_fn(item, item_options):
        return (np.array((1,)), np.array((2,)))

    with TemporaryDirectory() as tmp_dir:
        # Don't save to cache
        preprocess_items(
            dataset_name="test",
            dataset=mock_dataset,
            item_name="test",
            item_keys=item_keys,
            item_options={},
            preprocess_fn=preprocess_fn,
            load_from_cache=False,
            save_to_cache=False,
            cache_root=tmp_dir,
        )
        # Assert no cache files have been created
        assert len(list(Path(tmp_dir).glob("*"))) == 0

        # Save to cache
        preprocess_items(
            dataset_name="test",
            dataset=mock_dataset,
            item_name="test",
            item_keys=item_keys,
            item_options={},
            preprocess_fn=preprocess_fn,
            load_from_cache=False,
            save_to_cache=True,
            cache_root=tmp_dir,
        )
        # Assert one cache file has been created
        assert len(list(Path(tmp_dir).glob("*"))) == 1
        # Make sure number of files is correct
        assert len(list(Path(tmp_dir).glob("**/*.npy"))) == len(item_keys)

        # Create new dataset without new features
        mock_dataset_from_cache = GeneratedGraphData(total_num_graphs=100, nodes_per_graph=3, edges_per_graph=6)

        # Ensure cache is loaded
        def preprocess_fn(item, item_options):
            # Create assertion if function is run, ie. cache hasn't been loaded
            assert False

        preprocess_items(
            dataset_name="test",
            dataset=mock_dataset_from_cache,
            item_name="test",
            item_keys=item_keys,
            item_options={},
            preprocess_fn=preprocess_fn,
            load_from_cache=True,
            save_to_cache=True,
            cache_root=tmp_dir,
        )

        # Check loaded cache is the same as originally generated
        for dataset_idx in range(len(mock_dataset.dataset)):
            for key in mock_dataset.dataset[dataset_idx][0]:
                np.testing.assert_array_equal(
                    mock_dataset_from_cache.dataset[dataset_idx][0][key], mock_dataset.dataset[dataset_idx][0][key]
                )
