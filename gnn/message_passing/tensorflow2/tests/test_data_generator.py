# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

from data_utils.data_generators import PackedBatchGenerator


def test_packed_data_generator():
    n_packs_per_batch = 2
    max_graphs_per_pack = 3
    max_nodes_per_pack = 248
    max_edges_per_pack = 512

    # Feature and edge sizes for ogbg-molhiv dataset
    node_feature_size = 9
    edge_feature_size = 3

    pbg = PackedBatchGenerator(
        n_packs_per_batch=n_packs_per_batch,
        n_epochs=1,
        max_graphs_per_pack=max_graphs_per_pack,
        max_nodes_per_pack=max_nodes_per_pack,
        max_edges_per_pack=max_edges_per_pack,
        dataset_name="ogbg-molhiv",
        randomize=False,
    )
    ds = pbg.get_tf_dataset()
    all_batches = [batch for batch in ds]

    batch_tuple, ground_truth, sample_weights = all_batches[0]
    nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx = batch_tuple

    # Assert shapes:
    assert nodes.shape == (n_packs_per_batch, max_nodes_per_pack, node_feature_size)
    assert edges.shape == (n_packs_per_batch, max_edges_per_pack, edge_feature_size)
    assert receivers.shape == (n_packs_per_batch, max_edges_per_pack)
    assert senders.shape == (n_packs_per_batch, max_edges_per_pack)
    assert node_graph_idx.shape == (n_packs_per_batch, max_nodes_per_pack)
    assert edge_graph_idx.shape == (n_packs_per_batch, max_edges_per_pack)

    assert ground_truth.shape == (n_packs_per_batch, max_graphs_per_pack, 1)  # dummy_dimension (to match sigmoid)
    assert sample_weights.shape == (n_packs_per_batch, max_graphs_per_pack)

    # Binary labels for each graph in each pack in batch (with extra dummy dimension for sigmoid)
    expected_labels = np.array([[[0], [0], [0]], [[0], [0], [0]]], np.int32)
    np.testing.assert_array_equal(ground_truth, expected_labels)

    # 0 represents a masked graph, 1 an active graph
    expected_sample_weights = np.array([[1, 0, 0], [1, 1, 0]], np.int32)
    np.testing.assert_array_equal(sample_weights, expected_sample_weights)

    # Get all ground truths and sample masks as if it were validation/test
    ground_truth_all, include_sample_mask_all = pbg.get_ground_truth_and_masks()

    # -1 represents a masked graph
    expected_ground_truth = np.array([[0, -1, -1], [0, 0, -1]], np.int32)
    np.testing.assert_array_equal(ground_truth_all[0:2], expected_ground_truth)

    # False represents a masked graph, True an active graph
    expected_include_sample_mask = np.array([[True, False, False], [True, True, False]], np.bool)
    np.testing.assert_array_equal(include_sample_mask_all[0:2], expected_include_sample_mask)
