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

import logging
from pathlib import Path

import numpy as np
import os
import sys

from data_utils.generated_dataset_loader import generate_mock_graph_data
from data_utils.graphsage_dataset_loader import load_graphsage_data
from data_utils.graph_dataset import (
    GraphDataset,
    HeterogeneousGraphDataset,
    HomogeneousGraphDataset
)
from data_utils.ogb_dataset_loader import load_ogb_dataset
from data_utils.ogb_lsc_dataset_loader import load_ogb_lsc_mag_dataset
from utilities.constants import GraphType, Task
from utilities.options import ALLOWED_DATASET_TYPE


def check_data_exists(dataset_path, dataset_name):
    if not os.path.exists(dataset_path):
        logging.fatal("Data {} is not present at {}.".format(dataset_name, dataset_path))
        logging.fatal("Please download data using python3 data_utils/dataset_loader.py --dataset-name {} --data-path {}".format(dataset_name, dataset_path))
        sys.exit(-1)

def load_dataset(
        dataset_path,
        dataset_name,
        precalculate_first_layer,
        adjacency_dtype,
        features_dtype,
        labels_dtype,
        normalize_features=True,
        regenerate_cache=False,
        save_dataset_cache=True,
        pca_features_path=None
):

    preprocessed_file_path = Path(dataset_path).absolute().joinpath(
        f"{dataset_name}_preprocessed"
        f"_normalized_{normalize_features}"
        f"_precalc_{precalculate_first_layer}"
        f"_adjdtype_{adjacency_dtype.__name__}"
        f"_featdtype_{features_dtype.__name__}"
        f"_labelsdtype_{labels_dtype.__name__}"
        ".pickle.gz"
    )
    base_directory = Path(dataset_path).absolute().joinpath(
        f"{dataset_name}_preprocessed/"
    )
    if dataset_name != "ogbn-lsc-mag240":
        if preprocessed_file_path.is_file() and not regenerate_cache:
            logging.info(
                f"Loading {dataset_name} preprocessed dataset from"
                f" {preprocessed_file_path}. If this is not intended,"
                " either remove this file or run with argument"
                " `--regenerate-dataset-cache` set to True.")
            return GraphDataset.load_preprocessed_dataset(preprocessed_file_path)
    else:
        if base_directory.is_dir() and not regenerate_cache:
            return HomogeneousGraphDataset.load_preprocessed_mag240_dataset(base_directory)

    logging.info(f"Loading raw dataset...")

    if dataset_name == "generated":
        num_nodes, edges, features, labels, dataset_splits = generate_mock_graph_data()
        dataset = HomogeneousGraphDataset(
            dataset_name=dataset_name,
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.MULTI_CLASS_CLASSIFICATION,
            graph_type=GraphType.UNDIRECTED,
        )
        add_undirected_connections = True
    elif dataset_name == "ppi":
        check_data_exists(dataset_path, dataset_name)
        num_nodes, edges, features, labels, dataset_splits = load_graphsage_data(
            dataset_path, dataset_name)
        dataset = HomogeneousGraphDataset(
            dataset_name=dataset_name,
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.BINARY_MULTI_LABEL_CLASSIFICATION,
            graph_type=GraphType.UNDIRECTED,
        )
        add_undirected_connections = True
    elif dataset_name == "reddit":
        check_data_exists(dataset_path, dataset_name)
        num_nodes, edges, features, labels, dataset_splits = load_graphsage_data(
            dataset_path, dataset_name)
        dataset = HomogeneousGraphDataset(
            dataset_name=dataset_name,
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.MULTI_CLASS_CLASSIFICATION,
            graph_type=GraphType.UNDIRECTED,
        )
        add_undirected_connections = True
    elif dataset_name == "ogbn-arxiv":
        check_data_exists(dataset_path, dataset_name)
        num_nodes, edges, features, labels, dataset_splits = load_ogb_dataset(
            dataset_path, dataset_name)
        dataset = HomogeneousGraphDataset(
            dataset_name=dataset_name,
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.MULTI_CLASS_CLASSIFICATION,
            graph_type=GraphType.DIRECTED,
        )
        # Making this directed graph undirected improves accuracy
        add_undirected_connections = True
    elif dataset_name == "ogbn-products":
        check_data_exists(dataset_path, dataset_name)
        num_nodes, edges, features, labels, dataset_splits = load_ogb_dataset(
            dataset_path, "ogbn-products")
        dataset = HomogeneousGraphDataset(
            dataset_name=dataset_name,
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.MULTI_CLASS_CLASSIFICATION,
            graph_type=GraphType.UNDIRECTED,
        )
        add_undirected_connections = False
    elif dataset_name == "ogbn-mag":
        check_data_exists(dataset_path, dataset_name)
        num_nodes, edges, features, labels, dataset_splits = load_ogb_dataset(
            dataset_path,
            dataset_name
        )
        dataset = HeterogeneousGraphDataset(
            dataset_name=dataset_name,
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.MULTI_CLASS_CLASSIFICATION,
            graph_type=GraphType.DIRECTED,
            node_types=("paper", "author", "institution", "field_of_study"),
            node_types_missing_features=("author", "institution", "field_of_study"),
            node_types_missing_labels=("author", "institution", "field_of_study"),
            node_types_missing_dataset_splits=("author", "institution", "field_of_study"),
            edge_types=(
                ("author", "affiliated_with", "institution"),
                ("author", "writes", "paper"),
                ("paper", "cites", "paper"),
                ("paper", "has_topic", "field_of_study")
            )
        )
        add_undirected_connections = False
        # Dictionary describing the mapping between nodes and features to average
        # for missing features. Ordered dict ensures dependence on features preserved.
        feature_mapping = [
            ("author", {"feature": "paper",
                        "edge_list": ('author', 'writes', 'paper')}),
            ("institution", {"feature": "author",
                             "edge_list": ('author', 'affiliated_with', 'institution')})]
    elif dataset_name == "ogbn-lsc-mag240":
        check_data_exists(dataset_path, dataset_name)
        num_nodes, edges, features, labels, dataset_splits = load_ogb_lsc_mag_dataset(
            dataset_path,
            dataset_name,
            pca_features_path
        )
        logging.info("Processing MAG240 as heterogeneous dataset...")
        dataset = HeterogeneousGraphDataset(
            dataset_name=dataset_name,
            total_num_nodes=num_nodes,
            edges=edges,
            features=features,
            labels=labels,
            dataset_splits=dataset_splits,
            task=Task.MULTI_CLASS_CLASSIFICATION,
            graph_type=GraphType.DIRECTED,
            node_types=("paper", "author", "institution"),
            node_types_missing_features=(),  # Load from pca_features file, so no missing features
            node_types_missing_labels=("author", "institution"),
            node_types_missing_dataset_splits=(),
            edge_types=(
                ("author", "affiliated_with", "institution"),
                ("author", "writes", "paper"),
                ("paper", "cites", "paper")
            ))
        add_undirected_connections = False
        feature_mapping = None
    else:
        raise ValueError(f"Unrecognised dataset type: `{dataset_name}`."
                         f" Choose one of {ALLOWED_DATASET_TYPE}")
    logging.info(f"Raw dataset loaded.")

    logging.info(f"Preprocessing dataset...")

    if isinstance(dataset, HeterogeneousGraphDataset):
        if dataset_name in ["ogbn-lsc-mag240"]:
            dataset.generate_missing_labels()
        else:
            dataset.generate_missing_features(feature_mapping, dtype=np.float32)
            dataset.generate_missing_labels()
            dataset.generate_missing_dataset_splits()

        logging.info(f"Loaded heterogeneous dataset: {dataset}")

        dataset = dataset.to_homogeneous()

    dataset.generate_adjacency_matrices(adjacency_dtype)
    dataset.generate_masks()

    if normalize_features:
        dataset.normalize_features()
    if precalculate_first_layer:
        dataset.precalculate_first_layer()
    if add_undirected_connections:
        dataset.add_undirected_connections()
    dataset.remove_self_connections()

    dataset = dataset.features_to_dtype(features_dtype)
    dataset = dataset.labels_to_dtype(labels_dtype)

    logging.info(f"Dataset preprocessed.")

    if save_dataset_cache:
        if dataset_name not in ["generated", "ogbn-lsc-mag240"]:
            dataset.save(preprocessed_file_path)
        elif dataset_name == "ogbn-lsc-mag240":
            dataset.save_mag(base_directory)
    return dataset

if __name__ == "__main__":
    import argparse
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Cluster-GCN data downloader")
    parser.add_argument("--data-path",
                        type=str,
                        help="Path for the dataset.")
    parser.add_argument("--dataset-name",
                        type=str,
                        choices=ALLOWED_DATASET_TYPE,
                        help="Select dataset to use.")
    args = parser.parse_args()
    logging.info("Downloading {} to {}".format(args.dataset_name, args.data_path))
    if args.dataset_name in ["ppi", "reddit"]:
        load_graphsage_data(args.data_path, args.dataset_name)
    elif args.dataset_name in ["ogbn-arxiv", "ogbn-products", "ogbn-mag"]:
        load_ogb_dataset(args.data_path, args.dataset_name)
    elif args.dataset_name == "ogbn-lsc-mag240":
        pca_features_path = "/mag240m_kddcup2021/merged_feat_from_paper_feat_pca_129.npy"
        load_ogb_lsc_mag_dataset(args.data_path, args.dataset_name, pca_features_path)
