# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import copy
from collections import OrderedDict
import numpy as np
import pytest
import scipy.sparse as sp

from data_utils.dataset_loader import GraphDataset, HeterogeneousGraphDataset
from utilities.constants import MASKED_LABEL_VALUE, GraphType, Task


@pytest.mark.parametrize("in_data_dtype", [np.float16, np.float32])
def test_normalise_features(in_data_dtype):
    in_data = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        dtype=in_data_dtype,
    )
    normalize_by_entries = np.array([0, 1])

    expected_output = np.array(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0], [7.0, 7.0, 7.0], [9.0, 9.0, 9.0]],
        dtype=in_data_dtype,
    )

    output = GraphDataset.normalize(in_data, normalize_by_entries)

    np.testing.assert_almost_equal(output, expected_output)
    assert output.dtype == in_data_dtype


@pytest.mark.parametrize("in_data_dtype", [np.float16, np.float32])
def test_precalculate_first_layer_features(in_data_dtype):
    in_data = np.array([[0.01, 0.02], [0.11, 0.12]], dtype=in_data_dtype)
    adjacency = np.array([[0, 1], [1, 0]])

    expected_output = np.array([[0.11, 0.12, 0.01, 0.02], [0.01, 0.02, 0.11, 0.12]], dtype=in_data_dtype)

    output = GraphDataset.precalculate_first_layer_features(in_data, adjacency)

    np.testing.assert_almost_equal(output, expected_output)
    assert output.dtype == in_data_dtype


def test_remove_self_connections_from_adjacency():
    adjacency = sp.csr_matrix(
        [
            [1, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    expected_output = np.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )

    output = GraphDataset.remove_self_connections_from_adjacency(adjacency)

    np.testing.assert_almost_equal(output.toarray(), expected_output)
    assert adjacency.dtype == expected_output.dtype


def test_add_undirected_connections_to_adjacency():
    adjacency = sp.csr_matrix(
        [
            [1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    expected_output = np.array(
        [
            [1, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )

    output = GraphDataset.add_undirected_connections_to_adjacency(adjacency)

    np.testing.assert_almost_equal(output.toarray(), expected_output)
    assert adjacency.dtype == expected_output.dtype


@pytest.mark.parametrize("dtype", [np.bool, np.float16, np.float32, np.int32])
def test_convert_to_one_hot(dtype):
    dummy_labels = np.array([[0], [2], [4], [3], [2], [MASKED_LABEL_VALUE]])
    expected_labels = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [MASKED_LABEL_VALUE, MASKED_LABEL_VALUE, MASKED_LABEL_VALUE, MASKED_LABEL_VALUE, MASKED_LABEL_VALUE],
        ],
        dtype=dtype,
    )
    labels = GraphDataset.convert_to_one_hot(dummy_labels, dtype=dtype)
    np.testing.assert_array_equal(labels, expected_labels)
    assert dtype == labels.dtype


@pytest.mark.parametrize("dtype", [bool, np.float32])
def test_construct_adjacency(dtype):
    edge_list = np.array([[0, 4], [0, 3], [0, 1], [3, 4], [1, 2], [2, 5]])
    expected_full_adj_matrix = np.array(
        [
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    num_data = len(expected_full_adj_matrix)
    sparse_adj = GraphDataset.construct_adjacency(edge_list, num_data, dtype)
    adj = sparse_adj.toarray()
    np.testing.assert_array_equal(adj, expected_full_adj_matrix.astype(dtype))
    assert adj.dtype == dtype


def test_create_sample_mask():
    idx = np.array([1, 3, 4])
    num_nodes = 5
    expected_mask = np.array([0, 1, 0, 1, 1])

    mask = GraphDataset.create_sample_mask(idx, num_nodes)
    np.testing.assert_array_equal(mask, expected_mask)


def test_generate_train_edge_list():
    edge_list = np.array([[2, 4], [2, 1], [2, 0], [0, 1], [0, 3], [1, 4]])
    expected_train_edge_list = np.array([[2, 1], [2, 0], [0, 1]])
    dataset_splits = {"train": np.array([0, 1, 2]), "validation": np.array([3]), "test": np.array([4])}
    total_num_nodes = 5
    train_edge_list = GraphDataset.generate_train_edge_list(
        edge_list, total_num_nodes, dataset_splits["validation"], dataset_splits["test"]
    )
    np.testing.assert_array_equal(train_edge_list, expected_train_edge_list)


@pytest.mark.parametrize("in_data_dtype", [np.float16, np.float32])
def test_generate_train_features(in_data_dtype):
    features = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=in_data_dtype
    )
    expected_train_features = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=in_data_dtype
    )
    training_nodes = np.array([0, 1, 2])
    train_features = GraphDataset.generate_train_features(features, training_nodes)
    np.testing.assert_array_equal(train_features, expected_train_features)
    assert train_features.dtype == in_data_dtype


class TestHeterogeneousGraphDataset:
    @classmethod
    def setup_class(cls):
        num_nodes = {"x": 4, "y": 3, "z": 3}
        edges = {
            ("x", "to", "y"): np.array([[0, 0], [1, 2], [0, 1]]),
            ("y", "to", "z"): np.array([[0, 0], [1, 2], [0, 1]]),
            ("x", "to", "z"): np.array([[0, 0], [1, 2], [0, 1]]),
        }
        features = {
            "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            "y": np.array([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0]]),
        }
        labels = {
            "x": np.array([[1.0], [2.0], [3.0], [4.0]]),
            "y": np.array([[5.0], [6.0], [7.0]]),
        }
        dataset_splits = {
            "train": {
                "x": np.array([0, 1]),
                "y": np.array([0, 2]),
            },
            "validation": {
                "x": np.array([3]),
                "y": np.array([1]),
            },
            "test": {
                "x": np.array([2]),
                "y": np.array([], dtype=np.int32),
            },
        }
        cls.dataset = HeterogeneousGraphDataset(
            dataset_name="test",
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.MULTI_CLASS_CLASSIFICATION,
            graph_type=GraphType.DIRECTED,
            node_types=("x", "y", "z"),
            node_types_missing_features=("z"),
            node_types_missing_labels=("z"),
            node_types_missing_dataset_splits=("z"),
            edge_types=(("x", "to", "y"), ("y", "to", "z"), ("x", "to", "z")),
        )

    def test_generate_missing_labels(self):
        dataset = copy.copy(self.dataset)
        expected_labels = {
            "x": np.array([[1.0], [2.0], [3.0], [4.0]]),
            "y": np.array([[5.0], [6.0], [7.0]]),
            "z": np.array([[-1.0], [-1.0], [-1.0]]),
        }
        dataset.generate_missing_labels()

        for node_type in expected_labels:
            np.testing.assert_array_equal(dataset.labels[node_type], expected_labels[node_type])

    @pytest.mark.parametrize(
        "feat_mapping,z_feats",
        [
            ([], np.zeros((3, 3))),
            ([("z", {"feature": "x", "edge_list": ("x", "to", "z")})], np.array([[1.0, 2.0, 3.0], [5.5, 6.5, 7.5]])),
        ],
    )
    def test_generate_missing_features(self, feat_mapping, z_feats):
        dataset = copy.copy(self.dataset)
        feature_mapping = feat_mapping
        expected_features = {
            "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            "y": np.array([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0]]),
            "z": z_feats,
        }
        dataset.generate_missing_features(feature_mapping)

        for node_type in expected_features:
            np.testing.assert_array_equal(dataset.features[node_type], expected_features[node_type])

    def test_generate_missing_dataset_splits(self):
        dataset = copy.copy(self.dataset)
        expected_dataset_splits = {
            "train": {
                "x": np.array([0, 1]),
                "y": np.array([0, 2]),
                "z": np.array([0, 1, 2]),
            },
            "validation": {
                "x": np.array([3]),
                "y": np.array([1]),
                "z": np.array([]),
            },
            "test": {
                "x": np.array([2]),
                "y": np.array([]),
                "z": np.array([]),
            },
        }
        dataset.generate_missing_dataset_splits()

        for split, nodes_in_split in expected_dataset_splits.items():
            for node_type in nodes_in_split:
                np.testing.assert_array_equal(nodes_in_split[node_type], expected_dataset_splits[split][node_type])

    def test_to_homogeneous(self):
        dataset = copy.copy(self.dataset)
        expected_num_nodes = 10
        expected_edges = np.array([[0, 4], [1, 6], [0, 5], [4, 7], [5, 9], [4, 8], [0, 7], [1, 9], [0, 8]])
        expected_features = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        expected_labels = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [-1.0], [-1.0], [-1.0]])
        expected_dataset_splits = {
            "train": np.array([0, 1, 4, 6, 7, 8, 9]),
            "validation": np.array([3, 5]),
            "test": np.array([2]),
        }
        dataset.generate_missing_labels()
        dataset.generate_missing_features()
        dataset.generate_missing_dataset_splits()
        homogeneous_dataset = dataset.to_homogeneous()

        np.testing.assert_array_equal(homogeneous_dataset.total_num_nodes, expected_num_nodes)
        np.testing.assert_array_equal(homogeneous_dataset.edges, expected_edges)
        np.testing.assert_array_equal(homogeneous_dataset.features, expected_features)
        np.testing.assert_array_equal(homogeneous_dataset.labels, expected_labels)

        for split in expected_dataset_splits:
            np.testing.assert_array_equal(homogeneous_dataset.dataset_splits[split], expected_dataset_splits[split])
