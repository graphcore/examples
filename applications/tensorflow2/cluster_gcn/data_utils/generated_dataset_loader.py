# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np


def generate_random_dataset(num_nodes,
                            num_edges,
                            feature_size,
                            num_labels,
                            seed=10):
    """Generate random dataset to avoid using a real dataset."""
    np.random.seed(seed)
    # Splits in dataset:
    train_data = np.arange(num_nodes*0.8, dtype=np.int32)
    val_data = np.arange(num_nodes*0.8, num_nodes*0.9, dtype=np.int32)
    test_data = np.arange(num_nodes*0.9, num_nodes, dtype=np.int32)
    # Edge list:
    full_edges = np.random.randint(0, high=num_nodes, size=(num_edges, 2))
    # Features:
    full_features = np.ones((num_nodes, feature_size), dtype=np.float32)
    # Labels:
    full_labels = np.full((num_nodes, num_labels), 0, dtype=np.int32)

    return (num_nodes,
            full_edges,
            full_features,
            full_labels,
            train_data,
            val_data,
            test_data)


def generate_mock_graph_data():
    num_nodes = 5000
    num_edges = 81871
    feature_size = 50
    num_labels = 12
    return generate_random_dataset(num_nodes, num_edges, feature_size, num_labels)
