# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import scipy.sparse as sp

from data_utils.clustering_utils import partition_graph


def test_clustering_num_clusters():
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2]])
    num_nodes = 5
    adjacency = sp.csr_matrix(
        (
            np.ones((edge_list.shape[0]), dtype=np.float32),
            (edge_list[:, 0], edge_list[:, 1])
        ),
        shape=(num_nodes, num_nodes))
    adjacency += adjacency.transpose()

    idx_nodes = np.array([0, 1, 2, 3, 4])
    num_clusters = 2
    clusters_per_batch = 2

    graph_clusters, max_nodes_per_batch = partition_graph(
        adj=adjacency,
        directed_graph=False,
        idx_nodes=idx_nodes,
        num_clusters=num_clusters,
        clusters_per_batch=clusters_per_batch)

    assert sum([len(cluster) for cluster in graph_clusters]) == num_nodes
    assert len(graph_clusters) == num_clusters
    assert max_nodes_per_batch == 5


def test_clustering_single_cluster():
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2]])
    num_nodes = 5
    adjacency = sp.csr_matrix(
        (
            np.ones((edge_list.shape[0]), dtype=np.float32),
            (edge_list[:, 0], edge_list[:, 1])
        ),
        shape=(num_nodes, num_nodes))
    adjacency += adjacency.transpose()

    num_clusters = 1
    clusters_per_batch = 1
    idx_nodes = np.array([0, 1, 2, 3, 4])

    expected_clusters = [np.array([0, 1, 2, 3, 4])]

    graph_clusters, max_nodes_per_batch = partition_graph(
        adj=adjacency,
        directed_graph=False,
        idx_nodes=idx_nodes,
        num_clusters=num_clusters,
        clusters_per_batch=clusters_per_batch)

    assert sum([len(cluster) for cluster in graph_clusters]) == num_nodes
    assert max_nodes_per_batch == num_nodes
    np.testing.assert_array_equal(graph_clusters,
                                  expected_clusters)
