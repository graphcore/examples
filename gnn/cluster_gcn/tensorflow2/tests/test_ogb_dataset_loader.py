# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
from data_utils.graph_dataset import HeterogeneousGraphDataset
from utilities.constants import GraphType, Task
from pathlib import Path

from data_utils.ogb_dataset_loader import load_ogb_heterogeneous_dataset, load_ogb_homogeneous_dataset
from data_utils.ogb_lsc_dataset_loader import load_ogb_lsc_mag240_dataset
from data_utils.graph_dataset import GraphDataset


def test_load_homogeneous_ogb_dataset():
    graph = {
        "num_nodes": 5,
        "edge_index": np.array([[2, 2, 2, 0, 1], [4, 1, 0, 3, 4]]),
        "node_feat": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
    }
    labels = np.array([[2], [0], [3], [1], [4]])
    graph_split_idx = {"train": np.array([0, 1]), "valid": np.array([2]), "test": np.array([3, 4])}

    expected_edges = np.array([[2, 4], [2, 1], [2, 0], [0, 3], [1, 4]])
    expected_labels = np.array([[2.0], [0.0], [3.0], [1.0], [4.0]])
    expected_features = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]
    )

    (num_data, edges, features, labels, dataset_splits) = load_ogb_homogeneous_dataset(graph, labels, graph_split_idx)

    np.testing.assert_array_equal(edges, expected_edges)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(features, expected_features)

    assert num_data == len(features)
    assert num_data == len(labels)
    assert num_data == (len(dataset_splits["train"]) + len(dataset_splits["validation"]) + len(dataset_splits["test"]))


def test_load_heterogeneous_ogb_dataset():
    graph = {
        "num_nodes_dict": {"x": 5},
        "edge_index_dict": {("x", "to", "x"): np.array([[2, 2, 2, 0, 1], [4, 1, 0, 3, 4]])},
        "node_feat_dict": {"x": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])},
    }
    labels = {"x": np.array([[2], [0], [3], [1], [4]])}
    graph_split_idx = {"train": {"x": np.array([0, 1])}, "valid": {"x": np.array([2])}, "test": {"x": np.array([3, 4])}}

    expected_edges = {("x", "to", "x"): np.array([[2, 4], [2, 1], [2, 0], [0, 3], [1, 4]])}
    expected_labels = {"x": np.array([[2.0], [0.0], [3.0], [1.0], [4.0]])}
    expected_features = {
        "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    }

    (num_data, edges, features, labels, dataset_splits) = load_ogb_heterogeneous_dataset(graph, labels, graph_split_idx)

    for edge_type in expected_edges:
        np.testing.assert_array_equal(edges[edge_type], expected_edges[edge_type])

    for node_type, num_nodes in num_data.items():
        np.testing.assert_array_equal(labels[node_type], expected_labels[node_type])
        np.testing.assert_array_equal(features[node_type], expected_features[node_type])
        assert num_nodes == len(features[node_type])
        assert num_nodes == len(labels[node_type])
        assert num_nodes == (
            len(dataset_splits["train"][node_type])
            + len(dataset_splits["validation"][node_type])
            + len(dataset_splits["test"][node_type])
        )


def test_load_ogb_lsc_mag_dataset():
    class MockMAG(object):
        def __init__(self):
            author_institutiuon_edges = np.array([[0, 0], [1, 1], [2, 2]]).T
            author_paper_edges = np.array([[0, 0], [1, 1], [2, 2], [0, 3], [0, 4]]).T
            paper_paper_edges = np.array([[0, 0], [1, 1], [2, 2], [3, 4]]).T
            self.edge_index_dict = {
                ("author", "affiliated_with", "institution"): author_institutiuon_edges,
                ("author", "writes", "paper"): author_paper_edges,
                ("paper", "cites", "paper"): paper_paper_edges,
            }

        num_authors = 3
        num_papers = 5
        num_institutions = 3
        pca_feat = np.random.random_sample((11, 129))
        paper_label = np.random.randint(0, 10, size=(5,)).astype(np.float32)
        # Include a NaN for non-arXiv labeling
        paper_label[num_papers - 1] = np.nan

        def get_idx_split(self):
            return {
                "train": np.array([0, 1, 2]),
                "valid": np.array([0, 1, 2, 4]),
                "test-dev": np.array([0, 1, 2, 4, 5]),
            }

        def edge_index(self, str1, str2, str3):
            return self.edge_index_dict[(str1, str2, str3)]

    mock_mag240 = MockMAG()
    (num_data, edges, features, labels, dataset_splits) = load_ogb_lsc_mag240_dataset(mock_mag240)
    np.nan_to_num(mock_mag240.paper_label, nan=-1, copy=False)
    np.testing.assert_array_equal(labels["paper"], mock_mag240.paper_label[:, np.newaxis])
    np.testing.assert_array_equal(np.concatenate([v for _, v in features.items()]), mock_mag240.pca_feat)


def test_mag240_to_homogeneous():
    num_data = {"author": 3, "paper": 5, "institution": 3}
    edges = {
        ("author", "affiliated_with", "institution"): np.array([[0, 0], [1, 1], [2, 2]]),
        ("author", "writes", "paper"): np.array([[0, 0], [1, 1], [2, 2], [0, 3], [0, 4]]),
        ("paper", "cites", "paper"): np.array([[0, 0], [1, 1], [2, 2], [3, 4]]),
    }
    features = {
        "paper": np.random.random_sample((5, 129)),
        "author": np.random.random_sample((3, 129)),
        "institution": np.random.random_sample((3, 129)),
    }
    labels = {"paper": np.array([[0], [3], [9], [2], [-1]], dtype=np.int32)}
    dataset_splits = {
        "train": {
            "paper": np.array([0, 1, 2], dtype=np.int32),
            "author": np.array([0, 1, 2], dtype=np.int32),
            "institution": np.array([0, 1, 2], dtype=np.int32),
        },
        "validation": {
            "paper": np.array([0, 1, 2, 4], dtype=np.int32),
            "author": np.array([], dtype=np.int32),
            "institution": np.array([], dtype=np.int32),
        },
        "test": {
            "paper": np.array([0, 1, 2, 4, 5], dtype=np.int32),
            "author": np.array([], dtype=np.int32),
            "institution": np.array([], dtype=np.int32),
        },
    }

    dataset_name = "ogbn-lsc-mag240"
    dataset = HeterogeneousGraphDataset(
        dataset_name=dataset_name,
        total_num_nodes=num_data,
        edges=edges,
        features=features,
        labels=labels,
        dataset_splits=dataset_splits,
        task=Task.MULTI_CLASS_CLASSIFICATION,
        graph_type=GraphType.DIRECTED,
        node_types=("paper", "author", "institution"),
        node_types_missing_features=(),
        node_types_missing_labels=("author", "institution"),
        node_types_missing_dataset_splits=(),
        edge_types=(
            ("author", "affiliated_with", "institution"),
            ("author", "writes", "paper"),
            ("paper", "cites", "paper"),
        ),
        skip_train_feats_and_edges_allocation=True,
    )

    if isinstance(dataset, HeterogeneousGraphDataset):
        dataset.generate_missing_labels()
        dataset = dataset.to_homogeneous()

    dataset.generate_adjacency_matrices(np.bool)
    dataset.generate_masks()
    dataset.labels_to_one_hot(dtype=np.float32)
    dataset.normalize_features()
    dataset.precalculate_first_layer()
    dataset.add_undirected_connections()
    dataset.remove_self_connections()

    assert dataset.total_num_nodes == 11
    assert dataset.num_nodes == {"train": 9, "validation": 4, "test": 5}
    assert dataset.num_edges == {"train": 12, "validation": 12, "test": 12}

    base_directory = ".save_tmp/"
    dataset.save_mag(base_directory)

    loaded_dataset = GraphDataset.load_preprocessed_mag240_dataset(base_directory)

    assert dataset.num_edges == loaded_dataset.num_edges
