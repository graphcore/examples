# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import numpy as np
import tensorflow as tf

from data_utils.feature_generation import path_algorithms
from data_utils.feature_generation.utils import get_in_degrees


def get_centrality_encoding_from_dataset(dataset_item, item_options):
    return _preprocess_item_centrality_encoding(dataset_item["num_nodes"], dataset_item["edge_index"], **item_options)


def _preprocess_item_centrality_encoding(num_nodes, edges):
    centrality = get_in_degrees(edges, num_nodes)
    # Centrality 0 = non-trainable padding
    centrality += 1
    return (centrality,)


def get_shortest_path_distances_from_dataset(dataset_item, item_options):
    return _preprocess_item_shortest_path_distances(
        dataset_item["num_nodes"], dataset_item["edge_index"], **item_options
    )


def _preprocess_item_shortest_path_distances(num_nodes, edge_idxs, max_shortest_path_distance):
    senders, receivers = edge_idxs[0], edge_idxs[1]
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    # Force symmetric path-finding, i.e. bi-directional edges
    adj[senders, receivers] = True
    adj[receivers, senders] = True
    path_lengths, _ = path_algorithms.floyd_warshall(adj)
    path_lengths[path_lengths == 510] = -1  # 510 is magic value in floyd_warshall implying disconnected
    assert (
        np.max(path_lengths) <= max_shortest_path_distance
    ), f"Increase --model.max_shortest_path_distance to at least {np.max(path_lengths)}"

    # +1 makes space for value 0 to be non-trainable padding index in the embedding
    # Path lengths are used as categoricals, so offsetting by 1 doesn't change their function
    path_lengths += 1
    path_lengths = path_lengths.astype(np.int16)

    return (path_lengths,)


def get_send_rcv_from_dataset(dataset_item, item_options):
    return _preprocess_item_send_rcv(dataset_item["edge_feat"], dataset_item["edge_index"], **item_options)


def _preprocess_item_send_rcv(edge_feat, edge_idx, bidirectional=True):
    assert isinstance(bidirectional, bool)
    feat = edge_feat[::2, :]
    senders = edge_idx[0, ::2]
    receivers = edge_idx[1, ::2]
    if bidirectional:
        return (
            np.concatenate([feat, feat], axis=0),
            np.concatenate([senders, receivers], axis=-1),
            np.concatenate([receivers, senders], axis=-1),
        )
    else:
        return feat, senders, receivers


def get_graph_idxs_from_dataset(dataset_item, item_options):
    return _preprocess_item_graph_idxs(**item_options)


def _preprocess_item_graph_idxs():
    # Dummy graph indices for now, these are created properly in the batch generator
    node_graph_idxs = np.zeros(1, dtype=np.uint8)
    edge_graph_idxs = np.zeros(1, dtype=np.uint8)

    return node_graph_idxs, edge_graph_idxs


def _check_for_nans(tensor):
    return np.any(np.isnan(tensor))


def get_bond_lengths_from_dataset(dataset_item, item_options):
    nan_in_conformer = dataset_item.get("nan_in_conformer", _check_for_nans(dataset_item["ogb_conformer"]))
    return (
        _preprocess_item_bond_lengths(
            dataset_item["ogb_conformer"], dataset_item["senders"], dataset_item["receivers"], **item_options
        ),
        nan_in_conformer,
    )


def calculate_bond_lengths(pos, senders, receivers):
    edge_vec = pos[senders] - pos[receivers]
    edge_vec = edge_vec**2
    return np.sqrt(np.sum(edge_vec, axis=-1))


def _preprocess_item_bond_lengths(ogb_pos, senders, receivers, **kwargs):
    ogb_bond_lengths = calculate_bond_lengths(ogb_pos, senders, receivers)
    return ogb_bond_lengths


def get_atom_distances_from_dataset(dataset_item, item_options):
    nan_in_conformer = dataset_item.get("nan_in_conformer", _check_for_nans(dataset_item["ogb_conformer"]))
    return (*_preprocess_item_atom_distances(dataset_item["ogb_conformer"], **item_options), nan_in_conformer)


def _preprocess_item_atom_distances(ogb_pos):
    # Calculate direction vector based on the original atom positions
    # shape: n_nodes
    # convert to shape n_nodes, n_nodes
    relative_3D_pos = np.expand_dims(ogb_pos, axis=0) - np.expand_dims(ogb_pos, axis=1)

    ogb_atom_distances = np.linalg.norm(relative_3D_pos, axis=-1)
    # shape of direction vector: [num_nodes, num_nodes, 3]
    direction_vector = relative_3D_pos / (tf.expand_dims(ogb_atom_distances, axis=-1) + 1e-05)

    return (ogb_atom_distances, direction_vector)


def get_relative_features_from_dataset(dataset_item, item_options):
    return _preprocess_relative_features_from_dataset(dataset_item, **item_options)


def _preprocess_relative_features_from_dataset(dataset_item, relative_features_list, mode, size):
    senders = dataset_item["senders"]
    receivers = dataset_item["receivers"]
    if mode == "target_to_source":
        relative_feature_fn = get_target_to_source_relative_feature
    elif mode == "source_to_target":
        relative_feature_fn = get_source_to_target_relative_feature
    elif mode == "both":
        relative_feature_fn = get_relative_feature_from_both_direction
    elif mode == "abs":
        relative_feature_fn = get_absolute_relative_feature
    elif mode == "square":
        relative_feature_fn = get_squared_relative_feature
    else:
        raise ValueError(f"Mode {mode} not supported when getting relative feature.")

    relative_features = []
    if len(relative_features_list) > 0:
        for feature in relative_features_list:
            if feature == "random_walk":
                relative_random_walk = []
                for sender, receiver in zip(senders, receivers):
                    relative_random_walk += [
                        relative_feature_fn(dataset_item["random_walk_landing_probs"], sender, receiver)
                    ]
                # If there is just one atom in one molecule, there is no need to calculate
                # relative feature as there is no edge.
                if len(relative_random_walk) > 0:
                    relative_features += [np.stack(relative_random_walk)]
            elif feature == "laplacian_eig":
                # the eigen values for each node will be the same, so no need to take the relative value
                evects = dataset_item["lap_eig_vecs"]
                relative_eig_vec = []
                for sender, receiver in zip(senders, receivers):
                    relative_eig_vec += [relative_feature_fn(evects, sender, receiver)]
                # If there is just one atom in one molecule, there is no need to calculate
                # relative feature as there is no edge.
                if len(relative_eig_vec) > 0:
                    relative_features += [np.stack(relative_eig_vec)]
            else:
                raise ValueError(f"Relative feature {feature} not implemented.")

        if len(relative_features) == 0:
            # return a 2D tensor of zeros with the shape of [0, size] if there is no edge in one graph
            # because they will need to be able to concate with the other non-empty 2D  relative features
            return (np.zeros((0, size)),)
        else:
            output = np.concatenate(relative_features, axis=-1) if len(relative_features) > 1 else relative_features[0]
        output_size = len(output[0])
        # make sure the user input size is the same as the real relative feature size
        assert output_size == size
        return (output,)
    else:
        raise ValueError(
            f"Non-empty relative_features_list must be provided for the relative features. If not intend to use, remove the relative_feature argumment from config"
        )


def get_target_to_source_relative_feature(feature_from_dataset, sender, receiver):
    relative_feature = feature_from_dataset[receiver] - feature_from_dataset[sender]
    return relative_feature


def get_source_to_target_relative_feature(feature_from_dataset, sender, receiver):
    relative_feature = feature_from_dataset[sender] - feature_from_dataset[receiver]
    return relative_feature


def get_relative_feature_from_both_direction(feature_from_dataset, sender, receiver):
    relative_feature = np.concatenate(
        (
            get_target_to_source_relative_feature(feature_from_dataset, sender, receiver),
            get_source_to_target_relative_feature(feature_from_dataset, sender, receiver),
        )
    )
    return relative_feature


def get_absolute_relative_feature(feature_from_dataset, sender, receiver):
    relative_feature = np.abs(feature_from_dataset[sender] - feature_from_dataset[receiver])
    return relative_feature


def get_squared_relative_feature(feature_from_dataset, sender, receiver):
    relative_feature = np.square(feature_from_dataset[sender] - feature_from_dataset[receiver])
    return relative_feature


def trim_chemical_features(dataset_item, item_options):
    enforce_chemical_node_features_order = [
        "atomic_num",
        "chiral_tag",
        "degree",
        "possible_formal_charge",
        "possible_numH",
        "possible_number_radical_e",
        "possible_hybridization",
        "possible_is_aromatic",
        "possible_is_in_ring",
        "explicit_valence",
        "implicit_valence",
        "total_valence",
        "total_degree",
        "default_valence",
        "n_outer_electrons",
        "rvdw",
        "rb0",
        "env2",
        "env3",
        "env4",
        "env5",
        "env6",
        "env7",
        "env8",
        "gasteiger_charge",
        "donor",
        "acceptor",
        "num_chiral_centers",
    ]

    enforce_chemical_edge_features_order = [
        "possible_bond_type",
        "possible_bond_stereo",
        "possible_is_conjugated",
        "possible_is_in_ring",
        "possible_bond_dir",
    ]

    ordered_chemical_node_features = [
        k for k in enforce_chemical_node_features_order if k in item_options["chemical_node_features"]
    ]
    ordered_chemical_edge_features = [
        k for k in enforce_chemical_edge_features_order if k in item_options["chemical_edge_features"]
    ]

    logging.debug(f"Node features: {ordered_chemical_node_features}")
    logging.debug(f"Edge features: {ordered_chemical_edge_features}")

    index_to_keep = []
    for i in ordered_chemical_node_features:
        index_to_keep.append(enforce_chemical_node_features_order.index(i))
    new_node_feat_arr = []
    for j in dataset_item["node_feat"]:
        new_node_feat = []
        for idx in index_to_keep:
            if item_options["do_not_use_atomic_number"] and idx == 0:
                new_node_feat = new_node_feat
            else:
                new_node_feat.append(j[idx])
            if item_options["use_periods_and_groups"] and idx == 0:
                atom_row = row(j[idx] + 1)
                atom_group = group(j[idx] + 1)
                atom_family = family(j[idx] + 1)
                new_node_feat.extend([atom_row - 1, atom_group - 1, atom_family])
        new_node_feat_arr.append(new_node_feat)
    final_node_features = np.array(new_node_feat_arr)

    edge_index_to_keep = []
    for i in ordered_chemical_edge_features:
        edge_index_to_keep.append(enforce_chemical_edge_features_order.index(i))
    new_edge_feat_arr = []
    for j in dataset_item["edge_feat"]:
        new_edge_feat = []
        for idx in edge_index_to_keep:
            new_edge_feat.append(j[idx])
        new_edge_feat_arr.append(new_edge_feat)
    final_edge_features = np.array(new_edge_feat_arr)
    if final_edge_features.size == 0:
        final_edge_features = final_edge_features.reshape((0, len(item_options["chemical_edge_features"])))

    return (final_node_features, final_edge_features)


def row(z):
    _pt_row_sizes = (2, 8, 8, 18, 18, 32, 32)
    """
    Returns the periodic table row of the element.
    """
    total = 0
    if 57 <= z <= 71:
        return 8
    if 89 <= z <= 103:
        return 9
    for i, size in enumerate(_pt_row_sizes):
        total += size
        if total >= z:
            return i + 1
    return 8


def group(z):
    """
    Returns the periodic table group of the element.
    """
    if z == 1:
        return 1
    if z == 2:
        return 18
    if 3 <= z <= 18:
        if (z - 2) % 8 == 0:
            return 18
        if (z - 2) % 8 <= 2:
            return (z - 2) % 8
        return 10 + (z - 2) % 8

    if 19 <= z <= 54:
        if (z - 18) % 18 == 0:
            return 18
        return (z - 18) % 18

    if (z - 54) % 32 == 0:
        return 18
    if (z - 54) % 32 >= 18:
        return (z - 54) % 32 - 14
    if 57 <= z <= 71:
        return 3
    if 89 <= z <= 103:
        return 3
    return (z - 54) % 32


def family(z):
    """
    Returns the periodic table family of the element.
    0: Other non metals
    1: Alkali metals
    2: Alkaline earth metals
    3: Transition metals
    4: Lanthanides
    5: Actinides
    6: Other metals
    7: Metalloids
    8: Halogens
    9: Noble gases
    """
    if z in [1, 6, 7, 8, 15, 16, 34]:
        return 0
    if z in [3, 11, 19, 37, 55, 87]:
        return 1
    if z in [4, 12, 20, 38, 56, 88]:
        return 2
    if z in range(57, 72):
        return 4
    if z in range(89, 104):
        return 5
    if z in [13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 166]:
        return 6
    if z in [5, 14, 32, 33, 51, 52]:
        return 7
    if z in [9, 17, 35, 53, 85, 117]:
        return 8
    if z in [2, 10, 18, 36, 54, 86, 118]:
        return 9
    else:
        return 3
