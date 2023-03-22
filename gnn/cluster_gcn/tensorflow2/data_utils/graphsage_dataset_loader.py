# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging

import numpy as np


def load_graphsage_data(dataset_path, dataset_str):
    """
    Load GraphSAGE data from file.
    :param dataset_path: Path to the raw dataset.
    :param dataset_str: A string representing the name of Graphsage
        dataset.
    :return num_data: Integer representing the number of nodes in
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
    graph_file_name = f"{dataset_path}/{dataset_str}/{dataset_str}-G.json"
    logging.info(f"Loading graph data from {graph_file_name}...")
    with open(graph_file_name) as f:
        graph_json = json.load(f)

    nodes = graph_json["nodes"]
    graph_node_id_map = {i: node["id"] for i, node in enumerate(nodes)}

    orig_edges = graph_json["links"]

    id_map_file_name = f"{dataset_path}/{dataset_str}/{dataset_str}-id_map.json"
    logging.info(f"Loading id map from {id_map_file_name}...")
    with open(id_map_file_name) as f:
        id_map = json.load(f)
    is_digit = list(id_map.keys())[0].isdigit()
    id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}

    class_map_file_name = f"{dataset_path}/{dataset_str}/{dataset_str}-class_map.json"
    logging.info(f"Loading class map from {class_map_file_name}...")
    with open(class_map_file_name) as f:
        class_map = json.load(f)

    is_instance = isinstance(list(class_map.values())[0], list)
    class_map = {(int(k) if is_digit else k): (v if is_instance else int(v)) for k, v in class_map.items()}

    edges = []
    for edge in orig_edges:
        source_id = graph_node_id_map[edge["source"]]
        target_id = graph_node_id_map[edge["target"]]
        if source_id in id_map and target_id in id_map:
            edges.append((id_map[source_id], id_map[target_id]))
    edges = np.array(edges, dtype=np.int32)

    with open("{}/{}/{}-feats.npy".format(dataset_path, dataset_str, dataset_str), "rb") as f:
        features = np.load(f).astype(np.float32)

    num_data = len(id_map)
    train_data = np.array([id_map[n["id"]] for n in nodes if not n["val"] and not n["test"]])
    val_data = np.array([id_map[n["id"]] for n in nodes if n["val"]], dtype=np.int32)
    test_data = np.array([id_map[n["id"]] for n in nodes if n["test"]], dtype=np.int32)

    dataset_split = {"train": train_data, "validation": val_data, "test": test_data}

    # Process labels
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        labels = np.zeros((num_data, num_classes), dtype=np.int32)
        for k in class_map.keys():
            labels[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        labels = np.zeros((num_data, 1), dtype=np.int32)
        for k in class_map.keys():
            labels[id_map[k], 0] = class_map[k]

    return (num_data, edges, features, labels, dataset_split)
