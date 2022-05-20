# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging

import numpy as np
from ogb.nodeproppred import NodePropPredDataset


def get_labels_from_ogb_node_labels(node_labels):
    """Takes the preprocessed arxiv dataset node labels and returns a boolean array
    representing the labels, with a (boolean) one-hot encoding of the label assigned to
    that node."""
    node_labels = np.array(node_labels, dtype=np.int32)
    node_labels = node_labels.flatten()
    num_labels = max(node_labels) + 1
    num_nodes = len(node_labels)
    labels = np.zeros((num_nodes, num_labels), dtype=np.float32)
    for node, label in enumerate(node_labels):
        labels[node, label] = 1
    return labels


def load_ogb_dataset(dataset_name, dataset_path):
    """
    Load dataset from the Open Graph Benchmark python package.
    :param dataset_name: A string representing the name of the OGB
        dataset. This must match the dataset name of a dataset
        provided by the OGB python package.
    :param dataset_path: Path to cache the dataset.
    :return num_nodes: Integer representing the number of nodes in
        the dataset.
    :return edges:  Numpy array of edges, each value is a pair of node
        IDs which are connected.
    :return features: Numpy array of features, each value is a feature
        vector for that node index.
    :return labels: Numpy array of labels, each value is a label
        vector for that node index.
    :return train_data: Numpy array of node ids that are in the train
        dataset.
    :return val_data: Numpy array of node ids that are in the validation
        dataset.
    :return test_data: Numpy array of node ids that are in the test
        dataset.
    """
    logging.info(f"Loading OGB dataset {dataset_name}.")
    dataset = NodePropPredDataset(name=dataset_name, root=dataset_path)

    graph, labels = dataset[0]
    num_nodes = graph["num_nodes"]
    edges = graph["edge_index"].T

    features = np.array(graph["node_feat"], dtype=np.float32)
    assert num_nodes == len(features), (
        "There is a mismatch between the number of entries for"
        " node features and number of nodes. There should be one"
        " for each node.")

    full_labels = get_labels_from_ogb_node_labels(labels)
    assert num_nodes == len(full_labels), (
        "There is a mismatch between the number of entries for"
        " node labels and number of nodes. There should be one"
        " for each node.")

    split_idx = dataset.get_idx_split()
    train_ids = split_idx["train"]
    val_data = split_idx["valid"]
    test_data = split_idx["test"]
    assert num_nodes == len(train_ids) + len(val_data) + len(test_data), (
        "There is a mismatch between the number of entries for"
        " node dataset splits and number of nodes.")

    is_train = np.ones((num_nodes), dtype=np.bool)
    is_train[val_data] = False
    is_train[test_data] = False
    train_data = np.array([n for n in range(num_nodes) if is_train[n]],
                          dtype=np.int32)

    return (num_nodes,
            edges,
            features,
            full_labels,
            train_data,
            val_data,
            test_data)
