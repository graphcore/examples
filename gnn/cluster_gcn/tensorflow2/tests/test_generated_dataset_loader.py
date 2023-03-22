# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from data_utils.generated_dataset_loader import generate_random_dataset


def test_load_generated_dataset():
    num_nodes = 50000
    num_edges = 818716
    feature_size = 50
    num_labels = 121

    (dataset_num_nodes, dataset_edges, dataset_features, dataset_labels, dataset_splits) = generate_random_dataset(
        num_nodes, num_edges, feature_size, num_labels, seed=10
    )
    assert dataset_num_nodes == num_nodes
    assert len(dataset_edges.shape) == 2
    assert len(dataset_labels.shape) == 2
    assert len(dataset_features.shape) == 2
    assert dataset_edges.size > 2
    assert dataset_labels.size > 2
    assert dataset_features.size > 2
    assert num_nodes == (len(dataset_splits["train"]) + len(dataset_splits["validation"]) + len(dataset_splits["test"]))
