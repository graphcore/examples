# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging

import numpy as np
from ogb.nodeproppred import NodePropPredDataset


def load_ogb_heterogeneous_dataset(graph, labels, split_idx):
    num_nodes = graph["num_nodes_dict"]
    edges = {key: edge.T for key, edge in graph["edge_index_dict"].items()}
    features = {key: np.array(feat, dtype=np.float32) for key, feat in graph["node_feat_dict"].items()}
    labels = {key: np.array(lbl, dtype=np.int32) for key, lbl in labels.items()}
    dataset_split = {
        "train": {key: np.array(split, dtype=np.int32) for key, split in split_idx["train"].items()},
        "validation": {key: np.array(split, dtype=np.int32) for key, split in split_idx["valid"].items()},
        "test": {key: np.array(split, dtype=np.int32) for key, split in split_idx["test"].items()},
    }
    return (num_nodes,
            edges,
            features,
            labels,
            dataset_split)


def load_ogb_homogeneous_dataset(graph, labels, split_idx):
    num_nodes = graph["num_nodes"]
    edges = graph["edge_index"].T
    features = np.array(graph["node_feat"], dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    dataset_split = {
        "train": np.array(split_idx["train"], dtype=np.int32),
        "validation": np.array(split_idx["valid"], dtype=np.int32),
        "test": np.array(split_idx["test"], dtype=np.int32)
    }
    return (num_nodes,
            edges,
            features,
            labels,
            dataset_split)


def load_ogb_dataset(dataset_path, dataset_name):
    """
    Load dataset from the Open Graph Benchmark python package.
    :param dataset_path: Path to cache the dataset.
    :param dataset_name: A string representing the name of the OGB
        dataset. This must match the dataset name of a dataset
        provided by the OGB python package.
    :return num_nodes: Integer representing the number of nodes in
        the dataset.
    :return edges:  Numpy array of edges, each value is a pair of node
        IDs which are connected.
    :return features: Numpy array of features, each value is a feature
        vector for that node index.
    :return labels: Numpy array of labels, each value is a label
        vector for that node index.
    :return dataset_splits: Dictionary of numpy array of node ids that are in the train,
        validation and test datasets.
    """
    logging.info(f"Loading OGB dataset {dataset_name}.")
    dataset = NodePropPredDataset(name=dataset_name, root=dataset_path)

    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()

    if dataset.is_hetero:
        return load_ogb_heterogeneous_dataset(graph, labels, split_idx)
    else:
        return load_ogb_homogeneous_dataset(graph, labels, split_idx)
