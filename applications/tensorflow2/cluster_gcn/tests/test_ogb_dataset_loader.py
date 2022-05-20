# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np

from data_utils.ogb_dataset_loader import get_labels_from_ogb_node_labels


def test_get_labels_from_ogb_node_labels():
    dummy_labels = [[0], [2], [4], [3], [2]]
    expected_labels = np.array(
        [[1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0]]
    )
    labels = get_labels_from_ogb_node_labels(dummy_labels)
    np.testing.assert_array_equal(labels, expected_labels)
