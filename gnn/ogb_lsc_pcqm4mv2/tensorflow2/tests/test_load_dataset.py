# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

from data_utils.load_dataset import GeneratedOGBGraphData


def test_generated_data_loader():
    total_num_graphs = 100
    nodes_per_graph = 3
    edges_per_graph = 6
    node_feature_dims = np.array([119, 4, 12, 12, 10, 6, 6, 2, 2])
    edge_feature_dims = np.array([5, 6, 2])
    dataset = GeneratedOGBGraphData(
        total_num_graphs=total_num_graphs,
        node_feature_dims=node_feature_dims,
        edge_feature_dims=edge_feature_dims,
        nodes_per_graph=nodes_per_graph,
        edges_per_graph=edges_per_graph,
        labels_dtype=np.float32,
    )

    assert len(dataset.graphs) == total_num_graphs
    assert len(dataset.labels) == total_num_graphs
    for i in range(len(dataset.graphs)):
        graph = dataset.graphs[i]
        labels = dataset.labels[i]
        assert len(graph["edge_index"]) == 2
        assert len(graph["edge_index"][0]) == edges_per_graph
        assert len(graph["edge_feat"]) == edges_per_graph
        assert len(graph["edge_feat"][0]) == len(edge_feature_dims)
        assert len(graph["node_feat"]) == nodes_per_graph
        assert len(graph["node_feat"][0]) == len(node_feature_dims)
        assert graph["num_nodes"] == nodes_per_graph
        assert labels.shape == ()
