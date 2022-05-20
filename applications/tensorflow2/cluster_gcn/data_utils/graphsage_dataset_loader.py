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

from networkx.readwrite import json_graph
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
    graph_file_name = f'{dataset_path}/{dataset_str}/{dataset_str}-G.json'
    logging.info(f"Loading graph data from {graph_file_name}...")
    with open(graph_file_name) as f:
        graph_json = json.load(f)
    graph_nx = json_graph.node_link_graph(graph_json)

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
    class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
                 for k, v in class_map.items()}

    broken_count = 0
    to_remove = []
    for node in graph_nx.nodes():
        if node not in id_map:
            to_remove.append(node)
            broken_count += 1
    for node in to_remove:
        graph_nx.remove_node(node)
    logging.info(
        "Removed %d nodes that lacked proper annotations due to networkx versioning issues",
        broken_count)

    with open("{}/{}/{}-feats.npy".format(dataset_path, dataset_str, dataset_str), 'rb') as f:
        features = np.load(f).astype(np.float32)

    edges = []
    for edge in graph_nx.edges():
        if edge[0] in id_map and edge[1] in id_map:
            edges.append((id_map[edge[0]], id_map[edge[1]]))

    num_data = len(id_map)
    train_data = np.array([
        id_map[n]
        for n in graph_nx.nodes()
        if not graph_nx.node[n]["val"] and not graph_nx.node[n]["test"]
    ])
    val_data = np.array(
        [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]["val"]],
        dtype=np.int32)
    test_data = np.array(
        [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]["test"]],
        dtype=np.int32)

    # Process labels
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], class_map[k]] = 1

    logging.info(f"Raw dataset loaded.")

    return (num_data,
            edges,
            features,
            labels,
            train_data,
            val_data,
            test_data)
