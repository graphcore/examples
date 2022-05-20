# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

from data_utils.dataset_loader import Dataset


def test_normalise_features():
    in_data = np.array([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.],
                        [10., 11., 12.],
                        [13., 14., 15.],
                        [16., 17., 18.]])
    normalize_by_entries = np.array([0, 1])

    expected_output = np.array([[-1., -1., -1.],
                                [1., 1., 1.],
                                [3., 3., 3.],
                                [5., 5., 5.],
                                [7., 7., 7.],
                                [9., 9., 9.]])

    output = Dataset.normalize(in_data, normalize_by_entries)

    np.testing.assert_almost_equal(output, expected_output)


def test_precalculate_first_layer_features():
    in_data = np.array([[0.01, 0.02],
                        [0.11, 0.12]])
    adjacency = np.array([[0, 1],
                          [1, 0]])

    expected_output = np.array([[0.11, 0.12, 0.01, 0.02],
                                [0.01, 0.02, 0.11, 0.12]])

    output = Dataset.precalculate_first_layer_features(in_data, adjacency)

    np.testing.assert_almost_equal(output, expected_output)
