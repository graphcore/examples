# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

from data_utils.graphsage_dataset_loader import load_graphsage_data


def test_read_ppi_dataset(mocker):
    mock_graph = {
        "multigraph": False,
        "directed": False,
        "nodes": [
            {"id": 0, "test": False, "val": False},
            {"id": 1, "test": False, "val": False},
            {"id": 2, "test": False, "val": True},
            {"id": 3, "test": True, "val": False},
            {"id": 4, "test": True, "val": False},
        ],
        "links": [
            {"source": 0, "target": 4},
            {"source": 0, "target": 3},
            {"source": 0, "target": 1},
            {"source": 3, "target": 4},
            {"source": 1, "target": 2},
        ],
    }
    mock_id_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
    }
    mock_label_map = {
        "0": [0, 0, 1, 0, 0],
        "1": [0, 0, 1, 1, 0],
        "2": [0, 0, 0, 0, 1],
        "3": [0, 1, 1, 0, 0],
        "4": [1, 0, 0, 0, 0],
    }
    mock_features = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [10, 11, 12],
                              [13, 14, 15]])

    mocker.patch("builtins.open", mocker.mock_open(read_data=""))
    mocker.patch(
        "json.load",
        side_effect=(mock_graph, mock_id_map, mock_label_map)
    )
    mocker.patch("numpy.load", return_value=mock_features)

    expected_edges = np.array([[0, 4],
                               [0, 3],
                               [0, 1],
                               [3, 4],
                               [1, 2]])
    expected_labels = np.array([[0., 0., 1., 0., 0.],
                                [0., 0., 1., 1., 0.],
                                [0., 0., 0., 0., 1.],
                                [0., 1., 1., 0., 0.],
                                [1., 0., 0., 0., 0.]])
    expected_features = np.array([[1., 2., 3.],
                                  [4., 5., 6.],
                                  [7., 8., 9.],
                                  [10., 11., 12.],
                                  [13., 14., 15.]])

    (num_data,
     edges,
     features,
     labels,
     dataset_splits) = load_graphsage_data(".", "ppi")

    np.testing.assert_array_equal(edges, expected_edges)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(features, expected_features)

    assert num_data == len(features)
    assert num_data == len(labels)
    assert num_data == (len(dataset_splits["train"]) +
                        len(dataset_splits["validation"]) +
                        len(dataset_splits["test"]))


def test_read_reddit_dataset(mocker):
    mock_graph = {
        "multigraph": False,
        "directed": False,
        "nodes": [
            {"id": "ee", "test": False, "val": False},
            {"id": "aa", "test": False, "val": False},
            {"id": "bb", "test": False, "val": True},
            {"id": "cc", "test": True, "val": False},
            {"id": "dd", "test": True, "val": False},
        ],
        "links": [
            {"source": 0, "target": 4},
            {"source": 0, "target": 3},
            {"source": 0, "target": 1},
            {"source": 3, "target": 4},
            {"source": 1, "target": 2},
        ],
    }
    mock_label_map = {
        "ee": 1,
        "aa": 0,
        "bb": 3,
        "cc": 4,
        "dd": 2
    }
    mock_id_map = {
        "aa": 0,
        "cc": 1,
        "dd": 4,
        "bb": 3,
        "ee": 2,
    }
    mock_features = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [10, 11, 12],
                              [13, 14, 15]])

    mocker.patch("builtins.open", mocker.mock_open(read_data=""))
    mocker.patch(
        "json.load",
        side_effect=(mock_graph, mock_id_map, mock_label_map)
    )
    mocker.patch("numpy.load", return_value=mock_features)

    expected_edges = np.array([[2, 4],
                               [2, 1],
                               [2, 0],
                               [1, 4],
                               [0, 3]])
    expected_labels = np.array([[0.],
                                [4.],
                                [1.],
                                [3.],
                                [2.]])
    expected_features = np.array([[1., 2., 3.],
                                  [4., 5., 6.],
                                  [7., 8., 9.],
                                  [10., 11., 12.],
                                  [13., 14., 15.]])

    (num_data,
     edges,
     features,
     labels,
     dataset_splits) = load_graphsage_data(".", "reddit")

    np.testing.assert_array_equal(edges, expected_edges)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(features, expected_features)

    assert num_data == len(features)
    assert num_data == len(labels)
    assert num_data == (len(dataset_splits["train"]) +
                        len(dataset_splits["validation"]) +
                        len(dataset_splits["test"]))
