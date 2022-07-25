# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import math

import numpy as np
import pytest

from data_utils.clustering_utils import ClusterGraph
from utilities.constants import AdjacencyForm
from tests.utils import edge_list_to_sparse_adj


@pytest.mark.parametrize("adjacency_form", [
    AdjacencyForm.DENSE,
    AdjacencyForm.SPARSE_TENSOR,
    AdjacencyForm.SPARSE_TUPLE
])
@pytest.mark.parametrize("node_edge_imbalance_ratio", [
    None,
    [1.01, 1.11],
    [1.11, 1.01],
])
def test_max_nodes_per_batch_multiple_clusters(adjacency_form,
                                               node_edge_imbalance_ratio):
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2]])
    num_nodes = 5
    num_clusters = 2
    clusters_per_batch = 2
    adj = edge_list_to_sparse_adj(edge_list, num_nodes)

    graph_clusters = ClusterGraph(adjacency=adj,
                                  clusters_per_batch=clusters_per_batch,
                                  visible_nodes=range(num_nodes),
                                  num_clusters=num_clusters,
                                  directed_graph=True,
                                  adjacency_form=adjacency_form)
    graph_clusters.cluster_graph()
    assert graph_clusters.num_clusters == num_clusters
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        assert graph_clusters.max_nodes_per_batch == num_nodes + 1  # Fake node for edge padding.
    else:
        assert graph_clusters.max_nodes_per_batch == num_nodes


@pytest.mark.parametrize("adjacency_form", [
    AdjacencyForm.DENSE,
    AdjacencyForm.SPARSE_TENSOR,
    AdjacencyForm.SPARSE_TUPLE
])
def test_max_nodes_per_batch_single_cluster(adjacency_form):
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2]])
    num_nodes = 5
    num_clusters = 1
    clusters_per_batch = 1
    adj = edge_list_to_sparse_adj(edge_list, num_nodes)

    expected_clusters = np.array([[0, 1, 2, 3, 4]])

    graph_clusters = ClusterGraph(adjacency=adj,
                                  clusters_per_batch=clusters_per_batch,
                                  visible_nodes=range(num_nodes),
                                  num_clusters=num_clusters,
                                  directed_graph=True,
                                  adjacency_form=adjacency_form)
    graph_clusters.cluster_graph()
    assert len(graph_clusters.clusters[0]) == num_nodes
    np.testing.assert_array_equal(graph_clusters.clusters,
                                  expected_clusters)
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        assert graph_clusters.max_nodes_per_batch == num_nodes + 1  # Fake node for edge padding.
    else:
        assert graph_clusters.max_nodes_per_batch == num_nodes


def test_clustering_raises_value_error():
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2]])
    num_nodes = 5
    num_clusters = 2
    clusters_per_batch = 2
    adj = edge_list_to_sparse_adj(edge_list, num_nodes)
    with pytest.raises(ValueError):
        ClusterGraph(adjacency=adj,
                     clusters_per_batch=clusters_per_batch,
                     visible_nodes=range(num_nodes),
                     num_clusters=num_clusters,
                     max_nodes_per_batch=2,
                     directed_graph=True)


def test_num_clusters():
    num_clusters = ClusterGraph.get_num_clusters(num_nodes=1023,
                                                 max_nodes_per_batch=20,
                                                 clusters_per_batch=4)
    assert num_clusters == 205


@pytest.mark.parametrize("adjacency_form", [
    AdjacencyForm.DENSE,
    AdjacencyForm.SPARSE_TENSOR,
    AdjacencyForm.SPARSE_TUPLE
])
@pytest.mark.parametrize("inter_cluster_ratio", [0.0, 0.2])
def test_clustering_max_edges_per_batch(adjacency_form, inter_cluster_ratio):
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2],
                          [1, 5],
                          [2, 4],
                          [4, 6],
                          [3, 6]])
    num_nodes = 7
    clusters_per_batch = 2
    adj = edge_list_to_sparse_adj(edge_list, num_nodes)
    graph_clusters = ClusterGraph(adjacency=adj,
                                  clusters_per_batch=clusters_per_batch,
                                  visible_nodes=range(num_nodes),
                                  num_clusters=3,
                                  directed_graph=True,
                                  inter_cluster_ratio=inter_cluster_ratio,
                                  adjacency_form=adjacency_form)
    graph_clusters.cluster_graph()
    expected_num_edges = 3  # 1 edge in cluster [1, 5] and 2 in cluster [2, 4, 6]
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        expected_num_edges += math.ceil(inter_cluster_ratio * expected_num_edges)
        expected_num_edges += 6  # Self edges for 3 nodes in 2 clusters, plus fake node.

    assert graph_clusters.max_edges_per_batch == expected_num_edges


def test_cluster_graph_cache():
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2],
                          [1, 5],
                          [2, 4],
                          [4, 6],
                          [3, 6]])
    num_nodes = 7
    clusters_per_batch = 2
    adj = edge_list_to_sparse_adj(edge_list, num_nodes)
    graph_clusters = ClusterGraph(adjacency=adj,
                                  clusters_per_batch=clusters_per_batch,
                                  visible_nodes=range(num_nodes),
                                  num_clusters=3,
                                  directed_graph=True,
                                  adjacency_form=AdjacencyForm.DENSE)
    graph_clusters.cache_dir = "."
    graph_clusters.dataset_name = "test_clusters"
    graph_clusters.regenerate_cluster_cache = True
    # From scratch generate new clusters and save them
    graph_clusters.cluster_graph()
    original_clusters = graph_clusters._clusters.copy()
    # Reset the clusters and load from the file
    graph_clusters._clusters = None
    graph_clusters.regenerate_cluster_cache = False
    graph_clusters.cluster_graph()
    # assert that each cluster is the same as the original ones
    for x, y in zip(graph_clusters._clusters, original_clusters):
        np.testing.assert_equal(x, y)


def test_cluster_graph_cache_incorrect_file():
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2],
                          [1, 5],
                          [2, 4],
                          [4, 6],
                          [3, 6]])
    num_nodes = 7
    clusters_per_batch = 2
    adj = edge_list_to_sparse_adj(edge_list, num_nodes)
    graph_clusters = ClusterGraph(adjacency=adj,
                                  clusters_per_batch=clusters_per_batch,
                                  visible_nodes=range(num_nodes),
                                  num_clusters=3,
                                  directed_graph=True,
                                  adjacency_form=AdjacencyForm.DENSE)
    graph_clusters.cache_dir = "."
    # From scratch generate new clusters and save them
    graph_clusters.cluster_graph()
    original_clusters = graph_clusters._clusters.copy()

    # Clear the clusters and try to read from a file that doesn't exist
    # This will cluster but not save because regenerate is false
    graph_clusters._clusters = None
    graph_clusters.regenerate_cluster_cache = False  # Duplicated to be explicit
    graph_clusters.dataset_name = "fake_file_doesnt_exist"
    graph_clusters.cluster_graph()
    for x, y in zip(graph_clusters._clusters, original_clusters):
        np.testing.assert_equal(x, y)


def test_cluster_graph_cache_no_file():
    edge_list = np.array([[0, 4],
                          [0, 3],
                          [0, 1],
                          [3, 4],
                          [1, 2],
                          [1, 5],
                          [2, 4],
                          [4, 6],
                          [3, 6]])
    num_nodes = 7
    clusters_per_batch = 2
    adj = edge_list_to_sparse_adj(edge_list, num_nodes)
    graph_clusters = ClusterGraph(adjacency=adj,
                                  clusters_per_batch=clusters_per_batch,
                                  visible_nodes=range(num_nodes),
                                  num_clusters=3,
                                  directed_graph=True,
                                  adjacency_form=AdjacencyForm.DENSE)
    graph_clusters.regenerate_cluster_cache = False
    # From scratch generate new clusters and save them
    graph_clusters.cluster_graph()
    original_clusters = graph_clusters._clusters.copy()
    graph_clusters.dataset_name = "fake_file_doesnt_exist"

    # Then try to load the file that doesn't exist - check it hasn't been written
    graph_clusters._clusters = None

    edge_list += np.ones(edge_list.shape, dtype=np.int64)  # Change the edge list
    adj = edge_list_to_sparse_adj(edge_list, num_nodes+1)
    graph_clusters.adjacency = adj
    graph_clusters.cluster_graph()
    for x, y in zip(graph_clusters._clusters, original_clusters):
        # Hasn't loaded from the file (data change means we can test this)
        assert not np.array_equal(x, y)
