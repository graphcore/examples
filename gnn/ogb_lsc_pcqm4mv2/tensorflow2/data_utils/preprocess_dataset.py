# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_utils.feature_generation.generic_features import (
    get_shortest_path_distances_from_dataset,
    get_send_rcv_from_dataset,
    get_graph_idxs_from_dataset,
    get_relative_features_from_dataset,
    get_bond_lengths_from_dataset,
    get_centrality_encoding_from_dataset,
    get_atom_distances_from_dataset,
    trim_chemical_features,
)

from data_utils.feature_generation.laplacian_features import get_laplacian_features_from_dataset
from data_utils.feature_generation.random_walk_features import get_random_walk_landing_probs_from_dataset


def preprocess_dataset(dataset, options, load_ensemble_cache=True, folds=None, split_mode=None, ensemble=False):

    # For pad to max we don't do any trimming but must update any sizes of the inputs
    if not options.dataset.packing_strategy == "pad_to_max":

        k = [k for k in options.dataset.features.keys() if "senders_receivers" in k]
        assert len(k) == 1, f"only one sender_receiver feature should be included found: {k}"
        unidirectional = not options.dataset.features[k[0]].get("bidirectional", True)

        # must account for the fact the dataset assumes bidirectionality
        max_edges = 2 * options.model.n_edges_per_pack if unidirectional else options.model.n_edges_per_pack
        check_sizes(dataset, options.model.n_nodes_per_pack - 1, max_edges - 1, folds=folds)

    # make sure relative_feature is after senders_receivers, laplacian_eig and random_walk
    enforce_feature_order = [
        "senders_receivers",
        "bond_lengths",
        "laplacian_eig",
        "random_walk",
        "relative_feature",
    ]
    features = options.dataset.features.keys()
    ordered_features = [k for k in enforce_feature_order if k in features]
    unordered_features = [k for k in features if k not in enforce_feature_order]
    features = ordered_features + unordered_features
    logging.info(f"Preprocessing features in order: {features}")
    for feature in features:
        feature_options = options.dataset.features[feature]
        if feature == "chemical_features":
            item_keys = ("node_feat", "edge_feat")
            feature_options["chemical_node_features"] = options.dataset.chemical_node_features
            feature_options["use_periods_and_groups"] = options.dataset.use_periods_and_groups
            feature_options["do_not_use_atomic_number"] = options.dataset.do_not_use_atomic_number
            feature_options["chemical_edge_features"] = options.dataset.chemical_edge_features
            preprocess_fn = trim_chemical_features
        elif feature == "random_walk":
            item_keys = ("random_walk_landing_probs",)
            preprocess_fn = get_random_walk_landing_probs_from_dataset
        elif feature == "laplacian_eig":
            item_keys = ("lap_eig_vals", "lap_eig_vecs")
            preprocess_fn = get_laplacian_features_from_dataset
        elif feature == "centrality_encoding":
            item_keys = ("centrality_encoding",)
            preprocess_fn = get_centrality_encoding_from_dataset
        elif feature == "shortest_path_distances":
            item_keys = ("shortest_path_distances",)
            feature_options["max_shortest_path_distance"] = options.model.max_shortest_path_distance
            preprocess_fn = get_shortest_path_distances_from_dataset
        elif feature == "senders_receivers":
            item_keys = ("edge_feat", "senders", "receivers")
            preprocess_fn = get_send_rcv_from_dataset
        elif feature == "graph_idxs":
            item_keys = ("node_graph_idx", "edge_graph_idx")
            preprocess_fn = get_graph_idxs_from_dataset
        elif feature == "bond_lengths":
            item_keys = ("ogb_bond_lengths", "nan_in_conformer")
            feature_options["bidirectional"] = options.dataset.features["senders_receivers"]["bidirectional"]
            preprocess_fn = get_bond_lengths_from_dataset
        elif feature == "relative_feature":
            item_keys = ("relative_features",)
            preprocess_fn = get_relative_features_from_dataset
        elif feature == "atom_distance":
            item_keys = ("atom_distances", "direction_vector", "nan_in_conformer")
            preprocess_fn = get_atom_distances_from_dataset
        else:
            raise ValueError(f"Feature {feature} not supported in dataset preprocessing.")

        preprocess_items(
            dataset_name=options.dataset.dataset_name,
            dataset=dataset,
            item_name=feature,
            item_keys=item_keys,
            item_options=feature_options,
            preprocess_fn=preprocess_fn,
            load_from_cache=load_ensemble_cache if ensemble else options.dataset.load_from_cache,
            save_to_cache=options.dataset.save_to_cache,
            cache_root=options.dataset.cache_path,
            split_mode=split_mode,
            folds=folds,
        )

        check_index = dataset.check_idx[folds[0]] if folds is not None else 0
        for item_key in item_keys:
            assert item_key in dataset.dataset[check_index][0].keys()
    logging.info(f"Items in dataset: {dataset.dataset.graphs[0].keys()}")

    return dataset


def preprocess_items(
    dataset_name,
    dataset,
    item_name,
    item_options,
    item_keys,
    preprocess_fn,
    load_from_cache=False,
    save_to_cache=False,
    cache_root=Path("."),
    split_mode="original",
    folds=None,
):
    cache_path = get_cache_path(dataset_name, item_name, item_options, cache_root, split_mode, folds=folds)

    if load_from_cache:
        logging.info(f"Attempting to load preprocessed dataset item {item_name} from {cache_path}...")
        load_success = load_preprocessed_item(dataset, item_keys, cache_path)
        if load_success:
            logging.info(f"Successfully loaded preprocessed dataset item {item_name} from {cache_path}")
            return
        logging.info(f"Could not load preprocessed dataset item {item_name} from {cache_path}")

    # Do preprocessing
    for dataset_idx in tqdm(range(len(dataset.dataset)), desc=f"Generating {item_name} features..."):
        if folds is None or dataset.dataset_idx_in_splits(
            dataset_idx,
            folds,
        ):
            preprocessed_item = preprocess_fn(dataset.dataset[dataset_idx][0], item_options)
        else:
            preprocessed_item = [np.nan for _ in item_keys]

        assert isinstance(
            preprocessed_item, (tuple, list)
        ), f"Item returned from preprocessing function {preprocess_fn} must be a list."
        assert len(preprocessed_item) == len(item_keys), (
            f"Item returned {len(preprocessed_item)} from preprocessing function {preprocess_fn} must be the same"
            f" length as the provided item keys {len(item_keys)}."
        )
        dataset.dataset[dataset_idx][0].update(
            {item_key: preprocessed_item[item_key_idx] for item_key_idx, item_key in enumerate(item_keys)}
        )

    if save_to_cache:
        logging.info(f"Saving preprocessed item {item_name} to {cache_path}...")
        save_preprocessed_item(dataset, item_keys, cache_path)

    return


def hash_dict(in_dict):
    dhash = hashlib.md5()
    dhash.update(json.dumps(in_dict, sort_keys=True).encode())
    return dhash.hexdigest()


def get_cache_path(dataset_name, item_name, item_options, cache_root, split_mode=None, folds=None):
    cache_path = Path(cache_root)
    cache_path = cache_path.joinpath(f"{dataset_name}_preprocessed")
    # Save the cache for different split_mode and split_num and folds to different cache
    # folders.
    if split_mode is not None:
        cache_path = cache_path.joinpath(f"{split_mode}")
    if folds is not None:
        cache_path = cache_path.joinpath(f"{'_'.join(folds)}")
    cache_path = cache_path.joinpath(f"{item_name}_{hash_dict(item_options)}")
    return cache_path


def load_preprocessed_item(dataset, item_keys, cache_path):
    for item_key in item_keys:
        item_cache_path = cache_path.joinpath(f"{item_key}.npy")
        if item_cache_path.is_file():
            item_cache = np.load(item_cache_path, allow_pickle=True)
            for dataset_idx in range(len(dataset.dataset)):
                dataset.dataset[dataset_idx][0].update({item_key: item_cache[dataset_idx]})
        else:
            return False
    return True


def save_preprocessed_item(dataset, item_keys, cache_path):
    for item_key in item_keys:
        cache_path.mkdir(parents=True, exist_ok=True)
        item_cache_path = cache_path.joinpath(item_key)
        logging.info(f"Saving {item_key} to cache path {item_cache_path}...")
        item_list = np.array([dataset.dataset[dataset_idx][0][item_key] for dataset_idx in range(len(dataset.dataset))])
        np.save(item_cache_path, item_list)


def check_sizes(dataset, max_nodes, max_edges, folds=None):
    for dataset_idx in tqdm(range(len(dataset.dataset)), desc="Checking size of graphs against given options..."):
        if folds is None or dataset.dataset_idx_in_splits(dataset_idx, folds):
            num_edges = len(dataset.dataset[dataset_idx][0]["edge_index"][0])
            num_nodes = dataset.dataset[dataset_idx][0]["num_nodes"]
            if num_nodes > max_nodes or num_edges > max_edges:
                raise ValueError(
                    f"Found an item in the dataset nodes {num_nodes} and number of"
                    f" edges {num_edges}. This should be less than the selected"
                    " number of nodes/edges per pack, including one extra required"
                    " for padding."
                )
