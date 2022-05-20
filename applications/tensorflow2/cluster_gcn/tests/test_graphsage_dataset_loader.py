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
                               [1, 2],
                               [3, 4]])
    expected_labels = np.array([[0, 0, 1, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 0, 0, 0, 1],
                                [0, 1, 1, 0, 0],
                                [1, 0, 0, 0, 0]])
    expected_features = np.array([[1., 2., 3.],
                                  [4., 5., 6.],
                                  [7., 8., 9.],
                                  [10., 11., 12.],
                                  [13., 14., 15.]])

    (num_data,
     edges,
     features,
     labels,
     train_data,
     val_data,
     test_data) = load_graphsage_data(".", "ppi")

    np.testing.assert_array_equal(edges, expected_edges)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(features, expected_features)

    assert num_data == len(features)
    assert num_data == len(labels)
    assert num_data == len(train_data) + len(val_data) + len(test_data)


def test_read_reddit_dataset(mocker):
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
        mock_label_map = {
            "1": 0,
            "3": 1,
            "4": 4,
            "2": 3,
            "0": 2,
        }
        mock_id_map = {
            "0": 1,
            "1": 0,
            "2": 3,
            "3": 4,
            "4": 2
        }
        mock_features = np.array([[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9],
                                  [10, 11, 12],
                                  [13, 14, 15]])

        mocker.patch("builtins.open", mocker.mock_open(read_data=""))
        mocker.patch(
            "json.load",
            side_effect=(mock_graph, mock_label_map, mock_id_map)
        )
        mocker.patch("numpy.load", return_value=mock_features)

        expected_edges = np.array([[2, 4],
                                   [2, 1],
                                   [2, 0],
                                   [0, 3],
                                   [1, 4]])
        expected_labels = np.array([[1., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 1., 0., 0., 0.],
                                    [0., 0., 0., 1., 0.],
                                    [0., 0., 1., 0., 0.]])
        expected_features = np.array([[1., 2., 3.],
                                      [4., 5., 6.],
                                      [7., 8., 9.],
                                      [10., 11., 12.],
                                      [13., 14., 15.]])

        (num_data,
         edges,
         features,
         labels,
         train_data,
         val_data,
         test_data) = load_graphsage_data(".", "reddit")

        np.testing.assert_array_equal(edges, expected_edges)
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(features, expected_features)

        assert num_data == len(features)
        assert num_data == len(labels)
        assert num_data == len(train_data) + len(val_data) + len(test_data)
