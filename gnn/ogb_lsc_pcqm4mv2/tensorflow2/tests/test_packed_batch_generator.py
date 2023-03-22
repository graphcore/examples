# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf

from data_utils.load_dataset import GeneratedGraphData
from data_utils.packed_batch_generator import PackedBatchGenerator


def test_packed_batch_generator():
    n_packs_per_batch = 54
    n_graphs_per_pack = 3
    n_nodes_per_pack = 248
    n_edges_per_pack = 512

    # Feature and edge sizes for ogbg-molhiv dataset
    node_feature_size = 9
    edge_feature_size = 3

    input_spec = {
        "node_feat": {
            "shape": (n_nodes_per_pack + 1, node_feature_size),
            "input_name": "node_feat",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
        "edge_feat": {
            "shape": (n_edges_per_pack, edge_feature_size),
            "input_name": "edge_feat",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
        "node_graph_idx": {
            "shape": (n_nodes_per_pack + 1,),
            "input_name": "node_graph_idx",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
        "edge_graph_idx": {
            "shape": (n_edges_per_pack,),
            "input_name": "edge_graph_idx",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
    }

    dataset = GeneratedGraphData(total_num_graphs=2048, nodes_per_graph=24, edges_per_graph=50)

    pbg = PackedBatchGenerator(
        n_packs_per_batch=n_packs_per_batch,
        n_epochs=1,
        n_graphs_per_pack=n_graphs_per_pack,
        n_nodes_per_pack=n_nodes_per_pack,
        n_edges_per_pack=n_edges_per_pack,
        noisy_nodes_noise_prob=0.05,
        noisy_edges_noise_prob=0.05,
        dataset=dataset,
        randomize=False,
        input_spec=input_spec,
    )
    ds = pbg.get_tf_dataset(repeat_num=10)
    all_batches = [batch for batch in ds]
    batch, ground_truth = all_batches[0]
    maybe_noisy_nodes = batch["node_feat"]
    maybe_noisy_edges = batch["edge_feat"]
    node_graph_idx = batch["node_graph_idx"]
    edge_graph_idx = batch["edge_graph_idx"]

    # Assert shapes:
    assert maybe_noisy_nodes.shape == (n_packs_per_batch, n_nodes_per_pack + 1, node_feature_size)
    assert maybe_noisy_edges.shape == (n_packs_per_batch, n_edges_per_pack, edge_feature_size)
    assert node_graph_idx.shape == (n_packs_per_batch, n_nodes_per_pack + 1)
    assert edge_graph_idx.shape == (n_packs_per_batch, n_edges_per_pack)

    if type(ground_truth) is tuple:
        ground_truth = ground_truth[0]
    assert ground_truth.shape == (n_packs_per_batch, n_graphs_per_pack + 1, 1)  # dummy_dimension (to match sigmoid)

    # Binary labels for each graph in each pack in batch (with extra dummy dimension for sigmoid)
    for label in ground_truth:
        np.testing.assert_array_equal(label[0], np.array([-1]))

    # Get all ground truths and sample masks as if it were validation/test
    ground_truth_all, include_sample_mask_all = pbg.get_ground_truth_and_masks()

    # -1 represents a masked graph
    assert ground_truth_all.shape == (n_packs_per_batch, n_graphs_per_pack + 1)
    for label in ground_truth_all:
        np.testing.assert_array_equal(label[0], np.array(-1))

    # False represents a masked graph, True an active graph
    assert include_sample_mask_all.shape == (n_packs_per_batch, n_graphs_per_pack + 1)
    for mask in include_sample_mask_all:
        np.testing.assert_array_equal(mask[0], np.array(False))
