# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging

import numpy as np
from tqdm import tqdm


def append_to_batch_dict(
    batch_dict, this_graph, input_spec, n_nodes, n_edges, n_graphs, this_graph_n_nodes, this_graph_n_edges
):
    """Add the new items to the batch."""

    for inputs in input_spec.values():
        input_name = inputs["input_name"]
        if input_name == "node_graph_idx":
            batch_dict["node_graph_idx"].extend([n_graphs] * this_graph_n_nodes)
        elif input_name == "edge_graph_idx":
            # Extend the edge graph indexes by the number of edges in the current graph
            batch_dict["edge_graph_idx"].extend([n_graphs] * this_graph_n_edges)
        elif input_name in ("senders", "receivers"):
            # The node ids in the edges need to be offset by the number of nodes currently in the batch,
            # so that the node ids are relating to the correct nodes.
            batch_dict[input_name].append(this_graph[input_name] + n_nodes)
        elif input_name in (
            "node_feat",
            "edge_feat",
            "lap_eig_vals",
            "lap_eig_vecs",
            "random_walk_landing_probs",
            "centrality_encoding",
            "ogb_bond_lengths",
            "relative_features",
            "shortest_path_distances",
            "atom_distances",
            "direction_vector",
            "nan_in_conformer",
        ):
            # All features of the current graph just need to be added to the batch.
            batch_dict[input_name].append(this_graph[input_name])
        elif input_name in ("node_mask", "edge_mask"):
            pass
        else:
            raise NotImplementedError(f"Input name {input_name} has no implemented method for appending to batch dict.")

    return batch_dict


def pack_dataset(data_subset, max_nodes, max_edges, max_graphs, input_spec, silence_logging=False):
    # add padding node/graph
    max_nodes += 1
    max_edges += 1
    max_graphs += 1

    if not silence_logging:
        logging.info("Packing dataset...")

    def get_new_batch_dict():
        # initializes an empty batch dictionary

        _batch_dict = {}
        for inputs in input_spec.values():
            input_name = inputs["input_name"]
            input_data_type = inputs["input_dtype"].as_numpy_dtype
            input_pad_value = inputs["pad_value"]

            if input_name in ("senders", "receivers"):
                # For these inputs we need to ensure there is an extra item in the node dimension to account for a padding node
                first_item = [np.array([input_pad_value])]
            elif input_name in ("node_graph_idx", "edge_graph_idx", "distances", "max_path_length", "nan_in_conformer"):
                # For these inputs we need to ensure there is an extra item in the node dimension to account for a padding node
                first_item = [input_pad_value]
            elif input_name in (
                "node_feat",
                "edge_feat",
                "lap_eig_vals",
                "lap_eig_vecs",
                "random_walk_landing_probs",
                "centrality_encoding",
                "ogb_bond_lengths",
                "relative_features",
            ):
                # For these inputs we need to ensure there is an extra item in the node dimension to account for a padding node
                item_shape = list(data_subset[0][0][input_name].shape)
                node_dimension = 0
                item_shape[node_dimension] = 1
                first_item = [np.full(item_shape, input_pad_value, dtype=input_data_type)]
            elif input_name in ("direction_vector"):
                # 0 = non-trainable padding distance embedding, distinct from -1 = masked out attention
                first_item = [np.full((1, 1, 3), 0, dtype=input_data_type)]
            elif input_name in ("shortest_path_distances", "atom_distances"):
                # 0 = non-trainable padding distance embedding, distinct from -1 = masked out attention
                first_item = [np.full((1, 1), 0, dtype=input_data_type)]
            elif input_name in ("node_mask", "edge_mask"):
                pass
            else:
                raise NotImplementedError(
                    f"Input name {input_name} has no implemented method for creating a new batch."
                )

            _batch_dict[input_name] = first_item

        _batch_dict["labels"] = [-1]
        return _batch_dict

    packed_dataset = []

    batch_dict = get_new_batch_dict()

    all_n_nodes = []
    all_n_edges = []
    all_n_graphs = []

    pack_limit = {"nodes": 0, "edges": 0, "graphs": 0}

    if silence_logging:
        maybe_tqdm = lambda x: x
    else:
        maybe_tqdm = tqdm

    # node zero will be a dummy node that belongs to dummy graph 0
    n_nodes, n_edges, n_graphs = 1, 1, 1
    for this_graph, this_label in maybe_tqdm(data_subset):
        this_graph_n_nodes, this_graph_n_edges = this_graph["num_nodes"], this_graph["edge_feat"].shape[0]
        new_n_nodes = n_nodes + this_graph_n_nodes
        new_n_edges = n_edges + this_graph_n_edges
        new_n_graphs = n_graphs + 1

        pack_is_full = False
        if new_n_nodes > max_nodes:
            pack_is_full = True
            pack_limit["nodes"] += 1
        if new_n_edges > max_edges:
            pack_is_full = True
            pack_limit["edges"] += 1
        if new_n_graphs > max_graphs:
            pack_is_full = True
            pack_limit["graphs"] += 1

        if pack_is_full:
            all_n_nodes.append(n_nodes)
            all_n_edges.append(n_edges)
            all_n_graphs.append(n_graphs)

            packed_dataset.append(pad_graph(batch_dict, input_spec, max_graphs, max_nodes, max_edges))

            # initialize next batch dictionary
            batch_dict = get_new_batch_dict()

            batch_dict = append_to_batch_dict(
                batch_dict=batch_dict,
                this_graph=this_graph,
                input_spec=input_spec,
                n_nodes=1,
                n_edges=1,
                n_graphs=1,
                this_graph_n_nodes=this_graph_n_nodes,
                this_graph_n_edges=this_graph_n_edges,
            )
            batch_dict["labels"].append(this_label)
            # reinitialize
            n_nodes = this_graph_n_nodes + 1
            n_edges = this_graph_n_edges + 1
            # this_graph and the dummy graph
            n_graphs = 2

        else:
            batch_dict = append_to_batch_dict(
                batch_dict=batch_dict,
                this_graph=this_graph,
                input_spec=input_spec,
                n_nodes=n_nodes,
                n_edges=n_edges,
                n_graphs=n_graphs,
                this_graph_n_nodes=this_graph_n_nodes,
                this_graph_n_edges=this_graph_n_edges,
            )
            batch_dict["labels"].append(this_label)

            n_nodes = new_n_nodes
            n_edges = new_n_edges
            n_graphs = new_n_graphs

    all_n_nodes.append(n_nodes)
    all_n_edges.append(n_edges)
    all_n_graphs.append(n_graphs)
    packed_dataset.append(pad_graph(batch_dict, input_spec, max_graphs, max_nodes, max_edges))

    stats = {
        "avg_pack": {
            "nodes": np.mean(all_n_nodes) - 1,
            "edges": np.mean(all_n_edges) - 1,
            "graphs": np.mean(all_n_graphs) - 1,
        },
        "max_pack": {
            "nodes": np.max(all_n_nodes) - 1,
            "edges": np.max(all_n_edges) - 1,
            "graphs": np.max(all_n_graphs) - 1,
        },
        "pack_std": {"nodes": np.std(all_n_nodes), "edges": np.std(all_n_edges), "graphs": np.std(all_n_graphs)},
        "pack_limiter": {
            "nodes": pack_limit["nodes"] / len(all_n_nodes),
            "edges": pack_limit["edges"] / len(all_n_nodes),
            "graphs": pack_limit["graphs"] / len(all_n_nodes),
        },
        "pack_efficiency": {
            "nodes": (np.mean(all_n_nodes) - 1) / (max_nodes - 1),
            "edges": (np.mean(all_n_edges) - 1) / (max_edges - 1),
            "graphs": (np.mean(all_n_graphs) - 1) / (max_graphs - 1),
            "total": (np.mean(all_n_nodes) + np.mean(all_n_edges) - 1) / (max_nodes + max_edges - 1),
        },
    }
    if not silence_logging:
        logging.info("Finished packing dataset.")
    return packed_dataset, stats


def pad_graph(graph_dict, input_spec, n_graphs_post_padding, n_nodes_post_padding, n_edges_post_padding):
    """
    pads a graph to have a constant number of entries in all the fields (necessary for XLA
    compilation)
    :param graph_dict: a graph dictionary with keys n_nodes, node_features, edge_features,
                       edge_idx, labels
    :param n_graphs_post_padding: max number of samples in a batch
    :param n_nodes_post_padding:  max number of nodes in a batch (1 is reserved as a dummy node)
    :param n_edges_post_padding:  max number of edges in a batch
    :return: a graph dictionary:  the graph dictionary with its fields updated
    """
    padded_inputs = {}

    for input_s in input_spec.values():
        input_name = input_s["input_name"]
        inputs = graph_dict[input_name]
        padded_input = np.full(input_s["shape"], input_s["pad_value"], dtype=input_s["input_dtype"].as_numpy_dtype)
        if input_name in ("node_graph_idx", "edge_graph_idx", "nan_in_conformer"):
            # one-dimensional with scaler first value
            padded_input[: len(inputs)] = inputs
        elif input_name in ("edge_feat", "node_feat", "lap_eig_vecs", "random_walk_landing_probs", "relative_features"):
            # two-dimensional with numpy first value
            inputs = np.concatenate(inputs)
            padded_input[: len(inputs), :] = inputs
        elif input_name in ("centrality_encoding", "senders", "receivers", "ogb_bond_lengths"):
            inputs = np.concatenate(inputs)
            padded_input[: len(inputs)] = inputs
        elif input_name == "max_path_length":
            inputs = np.concatenate(inputs)
            padded_input[: len(inputs), : len(inputs), :, :] = inputs
        elif input_name == "lap_eig_vals":
            # three-dimensional
            inputs = np.concatenate(inputs)
            padded_input[: len(inputs), :, :] = inputs
        elif input_name == "direction_vector":
            pos = 0
            for distance_mat in graph_dict[input_name]:
                n = distance_mat.shape[0]
                padded_input[pos : pos + n, pos : pos + n, :] = distance_mat
                pos += n
            # Final rows are purely padding, will cause div-by-0 in softmax if all -inf, instead fill with the
            # 0 embedding index which is non-trainable but real-valued
            padded_input[pos:, pos:, :] = 0
        elif input_name in ("atom_distances", "shortest_path_distances"):
            pos = 0
            for distance_mat in graph_dict[input_name]:
                n = distance_mat.shape[0]
                padded_input[pos : pos + n, pos : pos + n] = distance_mat
                pos += n
            # Final rows are purely padding, will cause div-by-0 in softmax if all -inf, instead fill with the
            # 0 embedding index which is non-trainable but real-valued
            padded_input[pos:, pos:] = 0
        elif input_name in ("node_mask", "edge_mask"):
            pass
        else:
            raise NotImplementedError(f"Input name {input_name} has no implemented method for padding its data.")

        padded_inputs[input_name] = padded_input

    labels_array = -np.ones([n_graphs_post_padding], dtype=np.float32)
    labels_array[: len(graph_dict["labels"])] = graph_dict["labels"]
    padded_inputs["labels"] = labels_array

    return padded_inputs
