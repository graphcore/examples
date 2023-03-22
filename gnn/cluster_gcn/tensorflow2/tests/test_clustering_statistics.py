# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

from data_utils.clustering_statistics import ClusteringStatistics
from tests.utils import edge_list_to_sparse_adj


def get_example_adjacency():
    edge_list = np.array([[0, 4], [0, 3], [0, 1], [3, 4], [1, 2]])
    num_nodes = 5

    adj = edge_list_to_sparse_adj(edge_list, num_nodes)
    adj += adj.transpose()  # Test for undirected case.
    return adj


def test_get_sparsity_ratio():
    adjacency = get_example_adjacency()
    sparsity_ratio = ClusteringStatistics.get_sparsity_ratio(adjacency)
    assert sparsity_ratio == 0.6


def test_get_num_nodes_in_clusters():
    clusters = [np.array([1, 2]), np.array([0, 3, 4])]
    num_nodes = ClusteringStatistics.get_num_nodes_in_clusters(clusters)
    expected_num_nodes = [2, 3]
    assert expected_num_nodes == num_nodes


def test_get_cluster_degree():
    adjacency = get_example_adjacency()
    clusters = [np.array([1, 2]), np.array([0, 3, 4])]
    expected_cluster_degrees = [np.array([1, 1]), np.array([2, 2, 2])]

    for i in range(len(clusters)):
        cluster_degrees = ClusteringStatistics.get_cluster_degree(adjacency, clusters[i])
        np.testing.assert_array_almost_equal(cluster_degrees, expected_cluster_degrees[i])


def test_clustering_statistics():
    edge_list = np.array([[0, 4], [0, 3], [0, 1], [3, 4], [1, 2]])
    num_nodes = 5
    clusters_per_batch = 2

    adj = edge_list_to_sparse_adj(edge_list, num_nodes)
    adj += adj.transpose()  # Test for undirected case.

    clusters = [np.array([1, 2]), np.array([0, 3, 4])]

    clustering_statistics = ClusteringStatistics(adj, clusters, clusters_per_batch)
    clustering_statistics.evaluate_full_graph()
    clustering_statistics.evaluate_clustered_graph()
    clustering_statistics.evaluate_combined_clustered_graph()

    assert clustering_statistics.total_degree == 10
    assert clustering_statistics.total_cluster_degrees == 8
    assert clustering_statistics.total_combined_cluster_degrees == 10

    clustering_statistics.build_nx_graph_from_edge_list(clustering_statistics.adjacency_coo[0])
    full_graph_degrees = clustering_statistics.get_graph_degrees(clustering_statistics.full_graph)
    assert full_graph_degrees == [3, 2, 2, 2, 1]
