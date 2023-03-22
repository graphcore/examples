# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# This file has been modified by Graphcore Ltd.

import tensorflow as tf


def build_input_features(dataset, required_inputs, feature_to_input_spec, fold):

    selected_features = {}
    for input_name in required_inputs:
        if input_name in ["node_mask", "edge_mask"]:
            # features generated in the packed batch generator
            selected_features[input_name] = feature_to_input_spec[input_name]
        elif input_name not in dataset.dataset[dataset.check_idx[fold]][0]:
            raise ValueError(
                f"Feature {input_name} is not in the dataset, ensure it is included in the features required. Features available {dataset.dataset[0][0].keys()}"
            )
        elif input_name in feature_to_input_spec:
            selected_features[input_name] = feature_to_input_spec[input_name]
        else:
            raise ValueError(f"Key: `{input_name}` not currently supported.")
    return selected_features


def create_inputs_from_features(dataset, cfg, fold):
    # Feature Generation Dictionary:
    # Given all possible Features - and their metadata, pick based on config file
    # Update the feature_dict with required sizes

    if cfg.dataset.packing_strategy == "pad_to_max":
        n_graphs = dataset.stats[fold]["graphs"]["max"]
        n_nodes = dataset.stats[fold]["nodes"]["max"]
        n_edges = dataset.stats[fold]["edges"]["max"]
    else:
        n_graphs = cfg.model.n_graphs_per_pack
        n_nodes = cfg.model.n_nodes_per_pack
        n_edges = cfg.model.n_edges_per_pack

    n_nodes += 1  # includes padding
    n_graphs += 1  # includes padding
    n_edges += 1  # includes padding
    # the dummy node is the first of each pack
    dummy_node_idx = 0
    # the dummy edge is the first of each pack
    dummy_edge_idx = 0
    # the dummy graph is the first of each pack
    dummy_graph_idx = 0

    # if the first eigen value 0 is removed, the size of eigen vectors and eigen
    # values will be max_freqs - 1
    remove_first = cfg.dataset.features.get("laplacian_eig", {}).get("remove_first", None)
    if remove_first is not None:
        eig_shape = cfg.dataset.features.get("laplacian_eig", {}).get("max_freqs", None) - 1
    else:
        eig_shape = cfg.dataset.features.get("laplacian_eig", {}).get("max_freqs", None)

    feature_to_input_spec = {
        "node_feat": {
            "shape": (n_nodes, dataset.node_feature_size),
            "input_name": "node_feat",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
        "edge_feat": {
            "shape": (n_edges, dataset.edge_feature_size),
            "input_name": "edge_feat",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
        "centrality_encoding": {
            "shape": (n_nodes,),
            "input_name": "centrality_encoding",
            "model_dtype": tf.int32,
            "input_dtype": tf.uint8,
            "pad_value": 0,
        },
        "senders": {
            "shape": (n_edges,),
            "input_name": "senders",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": dummy_node_idx,
        },
        "receivers": {
            "shape": (n_edges,),
            "input_name": "receivers",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": dummy_node_idx,
        },
        "atom_distances": {
            "shape": (n_nodes, n_nodes),
            "input_name": "atom_distances",
            "model_dtype": tf.float32,
            "input_dtype": tf.float32,
            "pad_value": -1,
        },
        "direction_vector": {
            "shape": (n_nodes, n_nodes, 3),
            "input_name": "direction_vector",
            "model_dtype": tf.float32,
            "input_dtype": tf.float32,
            "pad_value": 0,
        },
        "max_path_length": {
            "shape": (n_nodes, n_nodes, cfg.model.max_path_length, dataset.edge_feature_size),
            "input_name": "path_feats",
            "model_dtype": tf.int32,
            "input_dtype": tf.uint8,
            "pad_value": 0,
        },
        "node_graph_idx": {
            "shape": (n_nodes,),
            "input_name": "node_graph_idx",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": dummy_graph_idx,
        },
        "edge_graph_idx": {
            "shape": (n_edges,),
            "input_name": "edge_graph_idx",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": dummy_graph_idx,
        },
        "shortest_path_distances": {
            "shape": (n_nodes, n_nodes),
            "input_name": "shortest_path_distances",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": -1,
        },
        "random_walk_landing_probs": {
            "shape": (n_nodes, len(cfg.dataset.features.get("random_walk", {}).get("k_steps", []))),
            "input_name": "random_walk_landing_probs",
            "model_dtype": tf.float32,
            "input_dtype": tf.float32,
            "pad_value": 0.0,
        },
        "lap_eig_vals": {
            "shape": (n_nodes, eig_shape, 1),
            "input_name": "lap_eig_vals",
            "model_dtype": tf.float32,
            "input_dtype": tf.float32,
            "pad_value": 0.0,
        },
        "lap_eig_vecs": {
            "shape": (n_nodes, eig_shape),
            "input_name": "lap_eig_vecs",
            "model_dtype": tf.float32,
            "input_dtype": tf.float32,
            "pad_value": 0.0,
        },
        "ogb_bond_lengths": {
            "shape": (n_edges,),
            "input_name": "ogb_bond_lengths",
            "model_dtype": tf.float32,
            "input_dtype": tf.float32,
            "pad_value": 0.0,
        },
        "relative_features": {
            "shape": (n_edges, cfg.dataset.features.get("relative_feature", {}).get("size")),
            "input_name": "relative_features",
            "model_dtype": tf.float32,
            "input_dtype": tf.float32,
            "pad_value": 0.0,
        },
        "node_mask": {
            "shape": (n_nodes,),
            "input_name": "node_mask",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
        "edge_mask": {
            "shape": (n_edges,),
            "input_name": "edge_mask",
            "model_dtype": tf.int32,
            "input_dtype": tf.int32,
            "pad_value": 0,
        },
        "nan_in_conformer": {
            "shape": (n_graphs,),
            "input_name": "nan_in_conformer",
            "model_dtype": tf.bool,
            "input_dtype": tf.bool,
            "pad_value": False,
        },
    }
    selected_input_features = build_input_features(dataset, cfg.inputs, feature_to_input_spec, fold)
    return selected_input_features
