# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np

import dataloader


def test_most_recent_indices() -> None:
    np.testing.assert_equal(
        dataloader.Data.most_recent_indices(np.array([10, 20, 30, 20, 30])),
        np.array([1, 0, 0, 1, 1], np.bool_),
    )
