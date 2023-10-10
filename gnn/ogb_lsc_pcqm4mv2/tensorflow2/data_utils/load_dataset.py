# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import rdkit
import tensorflow as tf
from ogb.graphproppred import GraphPropPredDataset
from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem import AllChem

from data_utils.pcq_dataset_28features import CustomPCQM4Mv2Dataset
from data_utils.pcq_dataset_28features import get_atom_feature_dims as get_atom_feature_dims_extended
from data_utils.pcq_dataset_28features import get_bond_feature_dims as get_bond_feature_dims_extended
from data_utils.pcq_dataset_28features import smiles2graph_large
import warnings
from tqdm import tqdm
import torch


def load_raw_dataset(
    dataset_name, dataset_cache_path, options, load_ensemble_cache=True, split=None, split_mode=None, ensemble=False
):
    if dataset_name == "generated":
        return GeneratedGraphData(
            total_num_graphs=options.dataset.generated_data_n_graphs,
            nodes_per_graph=options.dataset.generated_data_n_nodes,
            edges_per_graph=options.dataset.generated_data_n_edges,
            use_extended_features=False,
        )
    elif dataset_name == "generated_extended":
        return GeneratedGraphData(
            total_num_graphs=options.dataset.generated_data_n_graphs,
            nodes_per_graph=options.dataset.generated_data_n_nodes,
            edges_per_graph=options.dataset.generated_data_n_edges,
            use_extended_features=True,
            trim_chemical_features=options.dataset.trim_chemical_features,
            use_periods_and_groups=options.dataset.use_periods_and_groups,
            do_not_use_atomic_number=options.dataset.do_not_use_atomic_number,
            chemical_node_features=options.dataset.chemical_node_features,
            chemical_edge_features=options.dataset.chemical_edge_features,
        )
    elif dataset_name == "pcqm4mv2":
        return PCQM4Mv2GraphData(
            dataset_cache_path,
            use_extended_features=False,
            use_conformers=False,
            trim_chemical_features=options.dataset.trim_chemical_features,
            use_periods_and_groups=options.dataset.use_periods_and_groups,
            do_not_use_atomic_number=options.dataset.do_not_use_atomic_number,
            chemical_node_features=options.dataset.chemical_node_features,
            chemical_edge_features=options.dataset.chemical_edge_features,
            split=split,
            ensemble=ensemble,
            load_ensemble_cache=load_ensemble_cache,
            split_mode=split_mode if ensemble else options.dataset.split_mode,
            split_num=options.dataset.split_num,
            split_path=options.dataset.split_path,
        )
    elif dataset_name == "pcqm4mv2_conformers_28features":
        return PCQM4Mv2GraphData(
            dataset_cache_path,
            use_extended_features=True,
            use_conformers=True,
            num_processes=options.dataset.parallel_processes,
            trim_chemical_features=options.dataset.trim_chemical_features,
            use_periods_and_groups=options.dataset.use_periods_and_groups,
            do_not_use_atomic_number=options.dataset.do_not_use_atomic_number,
            chemical_node_features=options.dataset.chemical_node_features,
            chemical_edge_features=options.dataset.chemical_edge_features,
            split=split,
            ensemble=ensemble,
            load_ensemble_cache=load_ensemble_cache,
            split_mode=split_mode if ensemble else options.dataset.split_mode,
            split_num=options.dataset.split_num,
            split_path=options.dataset.split_path,
        )
    else:
        raise ValueError(f"Dataset name {dataset_name} not supported.")


class OGBGraphData:
    def __init__(
        self,
        use_extended_features=False,
        use_conformers=False,
        num_processes=240,
        trim_chemical_features=False,
        use_periods_and_groups=False,
        do_not_use_atomic_number=False,
        chemical_node_features=["atomic_num"],
        chemical_edge_features=["possible_bond_type"],
        split_mode="original",
        split_num=0,
        split_path="./pcqm4mv2-cross_val_splits/",
    ):
        self.dataset = {}
        self.labels_dtype = np.int32
        self.test_split_name = "test"
        self.task_score_mode = "max"
        self.num_processes = num_processes
        self.trim_chemical_features = trim_chemical_features
        self.use_periods_and_groups = use_periods_and_groups
        self.do_not_use_atomic_number = do_not_use_atomic_number
        self.chemical_node_features = chemical_node_features
        self.chemical_edge_features = chemical_edge_features
        self.split_mode = split_mode
        self.split_num = split_num
        self.split_path = split_path
        if use_extended_features:
            atom_features_dims = get_atom_feature_dims_extended()
            bond_features_dims = get_bond_feature_dims_extended()
        else:
            atom_features_dims = get_atom_feature_dims()
            bond_features_dims = get_bond_feature_dims()
        if self.trim_chemical_features:
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
                k for k in enforce_chemical_node_features_order if k in chemical_node_features
            ]
            index_to_keep = []
            atom_features_dims = []
            for i in ordered_chemical_node_features:
                # index_to_keep is not used here
                index_to_keep.append(enforce_chemical_node_features_order.index(i))
                if do_not_use_atomic_number and i == "atomic_num":
                    atom_features_dims = atom_features_dims
                else:
                    atom_features_dims.append(
                        get_atom_feature_dims_extended()[enforce_chemical_node_features_order.index(i)]
                    )
                if use_periods_and_groups and i == "atomic_num":
                    atom_features_dims.extend([9, 18, 10])

            ordered_chemical_edge_features = [
                k for k in enforce_chemical_edge_features_order if k in chemical_edge_features
            ]
            edge_index_to_keep = []
            bond_features_dims = []
            for i in ordered_chemical_edge_features:
                edge_index_to_keep.append(enforce_chemical_edge_features_order.index(i))
                bond_features_dims.append(
                    get_bond_feature_dims_extended()[enforce_chemical_edge_features_order.index(i)]
                )
        # atom_features_dims is a list with the atom features selected
        self.node_feature_dims = atom_features_dims
        self.node_feature_size = len(atom_features_dims)
        self.node_feature_sum = sum(atom_features_dims)
        self.node_feature_offsets = np.cumsum([0] + atom_features_dims[:-1])
        self.edge_feature_dims = bond_features_dims
        self.edge_feature_size = len(bond_features_dims)
        self.edge_feature_sum = sum(bond_features_dims)
        self.edge_feature_offsets = np.cumsum([0] + bond_features_dims[:-1])
        # WARNING: Only valid for pcq
        self.label_mean = 5.380503871833475
        self.label_std = 1.1785068841097899

        self.idx_to_split = None

    def normalize(self, values_to_normalize):
        return (values_to_normalize - self.label_mean) / self.label_std

    def denormalize(self, values_to_denormalize):
        return (values_to_denormalize * self.label_std) + self.label_mean

    def get_split(self, split_name):
        return [self.dataset[idx] for idx in self.dataset.split_dict[split_name]]

    def dataset_idx_in_splits(self, idx, splits):
        if self.idx_to_split is None:
            self.idx_to_split = [None] * len(self.dataset.graphs)
            for split_key, split_val in self.dataset.split_dict.items():
                for graph_idx in split_val:
                    self.idx_to_split[graph_idx] = split_key
            assert None not in self.idx_to_split
        return self.idx_to_split[idx] in splits

    def get_statistics(self, split=None, do_check_for_nans=False):
        stats = {}
        for split_name in self.dataset.split_dict:
            if split is None or split_name in split:
                if do_check_for_nans:
                    check_for_nans = ["labels", "node_feat", "edge_feat", "ogb_bond_lengths"]
                    nan_check = {k: [] for k in check_for_nans}
                else:
                    nan_check = {}
                n_edges = []
                n_nodes = []
                labels = []
                stats_for_split = {}
                for idx in tqdm(self.dataset.split_dict[split_name], desc=f"Getting stats for {split_name} split"):
                    item = self.dataset[idx][0]
                    label = self.dataset[idx][1]
                    n_edges.append(len(item["edge_feat"]))
                    n_nodes.append(item["num_nodes"])
                    if do_check_for_nans:
                        for k in nan_check.keys():
                            if k == "labels":
                                nan_check[k].append(np.isnan(label).any())
                            elif k in item.keys():
                                nan_check[k].append(np.isnan(item[k]).any())
                    labels.append(label)
                stats_for_split["nodes"] = {
                    "max": max(n_nodes),
                    "min": min(n_nodes),
                    "median": np.median(n_nodes),
                    "mean": np.mean(n_nodes),
                }
                stats_for_split["edges"] = {
                    "max": max(n_edges),
                    "min": min(n_edges),
                    "median": np.median(n_edges),
                    "mean": np.mean(n_edges),
                }
                if do_check_for_nans:
                    stats_for_split["nan_check"] = {
                        k: any(nan_check[k]) for k in nan_check.keys() if nan_check[k] != []
                    }

                stats[split_name] = stats_for_split
                logging.info(f"Dataset stats for split {split_name}: {stats[split_name]}")
        return stats

    def get_conformer_statistics(self):
        stats = {}
        for split_name in self.dataset.split_dict:
            ogb_bond_lengths = []
            atom_distances = []
            stats_for_split = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for idx in tqdm(
                    self.dataset.split_dict[split_name], desc=f"Getting conformer stats for {split_name} split"
                ):
                    item = self.dataset[idx][0]
                    ogb_bond_lengths.append(item.get("ogb_bond_lengths", [np.nan]))
                    atom_distances.append(item.get("atom_distances", [np.nan]).flatten())
                ogb_bond_lengths = np.concatenate(ogb_bond_lengths)
                atom_distances = np.concatenate(atom_distances)
                stats_for_split["conformers"] = {
                    "ogb BL mean": np.nanmean(ogb_bond_lengths),
                    "ogb BL stdev": np.nanstd(ogb_bond_lengths),
                    "atom_distance mean": np.nanmean(atom_distances),
                    "atom_distance stdev": np.nanstd(atom_distances),
                }
            stats[split_name] = stats_for_split
            logging.info(f"Dataset stats for split {split_name}: {stats[split_name]}")
        return stats

    def get_num_graphs_per_split(self):
        return {split_name: len(split) for split_name, split in self.dataset.split_dict.items()}


class PCQM4Mv2GraphData(OGBGraphData):
    def __init__(
        self,
        dataset_cache_path=Path("."),
        use_extended_features=True,
        use_conformers=False,
        num_processes=240,
        trim_chemical_features=False,
        use_periods_and_groups=False,
        do_not_use_atomic_number=False,
        chemical_node_features=["atomic_num"],
        chemical_edge_features=["possible_bond_type"],
        split=None,
        ensemble=False,
        load_ensemble_cache=True,
        split_mode="original",
        split_num=0,
        split_path="./pcqm4mv2-cross_val_splits/",
    ):
        super().__init__(
            use_extended_features=use_extended_features,
            use_conformers=use_conformers,
            num_processes=num_processes,
            trim_chemical_features=trim_chemical_features,
            use_periods_and_groups=use_periods_and_groups,
            do_not_use_atomic_number=do_not_use_atomic_number,
            chemical_node_features=chemical_node_features,
            chemical_edge_features=chemical_edge_features,
            split_mode=split_mode,
            split_num=split_num,
            split_path=split_path,
        )

        if ensemble:
            # During inference we would like to use only sub-set of the whole dataset,
            # the split and ensemble flags would help us differentiate that.
            if use_extended_features:
                smiles2graph_func = smiles2graph_large
            else:
                smiles2graph_func = smiles2graph
            self.dataset = CustomPCQM4Mv2Dataset(
                root=dataset_cache_path,
                smiles2graph=smiles2graph_func,
                use_extended_features=use_extended_features,
                use_conformers=use_conformers,
                num_processes=num_processes,
                trim_chemical_features=trim_chemical_features,
                use_periods_and_groups=use_periods_and_groups,
                do_not_use_atomic_number=do_not_use_atomic_number,
                chemical_node_features=chemical_node_features,
                chemical_edge_features=chemical_edge_features,
                split=split,
                ensemble=ensemble,
                load_ensemble_cache=load_ensemble_cache,
                split_mode=split_mode,
                split_num=split_num,
                split_path=split_path,
            )

        else:
            if use_extended_features and use_conformers:
                self.dataset = CustomPCQM4Mv2Dataset(
                    root=dataset_cache_path,
                    smiles2graph=smiles2graph_large,
                    use_extended_features=use_extended_features,
                    use_conformers=use_conformers,
                    num_processes=num_processes,
                    trim_chemical_features=trim_chemical_features,
                    use_periods_and_groups=use_periods_and_groups,
                    do_not_use_atomic_number=do_not_use_atomic_number,
                    chemical_node_features=chemical_node_features,
                    chemical_edge_features=chemical_edge_features,
                    split_mode=split_mode,
                    split_num=split_num,
                    split_path=split_path,
                )
            elif use_conformers and not use_extended_features:
                self.dataset = CustomPCQM4Mv2Dataset(
                    root=dataset_cache_path,
                    smiles2graph=smiles2graph,
                    use_extended_features=use_extended_features,
                    use_conformers=use_conformers,
                    num_processes=num_processes,
                    trim_chemical_features=trim_chemical_features,
                    use_periods_and_groups=use_periods_and_groups,
                    do_not_use_atomic_number=do_not_use_atomic_number,
                    chemical_node_features=chemical_node_features,
                    chemical_edge_features=chemical_edge_features,
                    split_mode=split_mode,
                    split_num=split_num,
                    split_path=split_path,
                )
            elif use_extended_features and not use_conformers:
                self.dataset = CustomPCQM4Mv2Dataset(
                    root=dataset_cache_path,
                    smiles2graph=smiles2graph_large,
                    use_extended_features=use_extended_features,
                    use_conformers=use_conformers,
                    num_processes=num_processes,
                    trim_chemical_features=trim_chemical_features,
                    use_periods_and_groups=use_periods_and_groups,
                    do_not_use_atomic_number=do_not_use_atomic_number,
                    chemical_node_features=chemical_node_features,
                    chemical_edge_features=chemical_edge_features,
                    split=split,
                    ensemble=ensemble,
                    split_mode=split_mode,
                    split_num=split_num,
                    split_path=split_path,
                )
            else:
                self.dataset = PCQM4Mv2Dataset(root=dataset_cache_path, smiles2graph=smiles2graph)

        if split_mode == "incl_half_valid":
            split_file = split_path + "incl_half_valid/split_dict_" + str(split_num) + ".pt"
            logging.info(f"Split file: {split_file}")
            self.dataset.split_dict = torch.load(split_file)
        elif split_mode == "47_kfold":
            split_file = split_path + "47_kfold/split_dict_" + str(split_num) + ".pt"
            logging.info(f"Split file: {split_file}")
            self.dataset.split_dict = torch.load(split_file)
        elif split_mode == "train_plus_valid":
            split_file = split_path + "train_plus_valid/split_dict.pt"
            logging.info(f"Split file: {split_file}")
            self.dataset.split_dict = torch.load(split_file)
        else:
            self.dataset.split_dict = self.dataset.get_idx_split()
            logging.info(f"Original dataset split")

        self.labels_dtype = np.float32
        self.tf_labels_dtype = tf.float32
        self.test_split_name = "test"
        self.task_score_mode = "min"
        self.total_num_graphs = len(self.dataset.graphs)
        self.total_num_graphs_per_split = self.get_num_graphs_per_split()
        self.stats = self.get_statistics(split=split)
        # The numbers for the check index are the first index in the corresponding split.
        # PCQ dataset only
        self.check_idx = {"train": 0, "valid": 3378606, "test-dev": 3378608, "test-challenge": 3378615}


class GeneratedGraphData(OGBGraphData):
    """
    This class makes randomly generated graph data for benchmarking
    these graphs are all the same size (no padding/dummy graphs) but
    it is still indicative of real performance
    """

    def __init__(
        self,
        total_num_graphs,
        nodes_per_graph,
        edges_per_graph,
        use_extended_features=False,
        trim_chemical_features=False,
        use_periods_and_groups=False,
        do_not_use_atomic_number=False,
        chemical_node_features=["atomic_num"],
        chemical_edge_features=["possible_bond_type"],
    ):
        super().__init__(
            use_extended_features=use_extended_features,
            trim_chemical_features=trim_chemical_features,
            use_periods_and_groups=use_periods_and_groups,
            do_not_use_atomic_number=do_not_use_atomic_number,
            chemical_node_features=chemical_node_features,
            chemical_edge_features=chemical_edge_features,
        )
        self.labels_dtype = np.float32
        self.tf_labels_dtype = tf.float32
        if use_extended_features:
            atom_features_dims = get_atom_feature_dims_extended()
            bond_features_dims = get_bond_feature_dims_extended()
        else:
            atom_features_dims = get_atom_feature_dims()
            bond_features_dims = get_bond_feature_dims()
        self.dataset = GeneratedOGBGraphData(
            total_num_graphs,
            atom_features_dims,
            bond_features_dims,
            nodes_per_graph,
            edges_per_graph,
            self.labels_dtype,
        )
        self.test_split_name = "valid"
        self.task_score_mode = "min"
        self.total_num_graphs = len(self.dataset.graphs)
        self.total_num_graphs_per_split = self.get_num_graphs_per_split()
        self.stats = self.get_statistics()
        self.check_idx = {"train": 0, "valid": 0, "test-dev": 0, "test-challenge": 0}


class GeneratedOGBGraphData:
    def __init__(
        self, total_num_graphs, node_feature_dims, edge_feature_dims, nodes_per_graph, edges_per_graph, labels_dtype
    ):
        assert edges_per_graph % 2 == 0, "Generated edges per graph must be a multiple of 2."
        num_node_feats = len(node_feature_dims)
        num_edge_feats = len(edge_feature_dims)
        np.random.seed(23)
        self.total_num_graphs = total_num_graphs
        self.graphs = [
            {
                "num_nodes": nodes_per_graph,
                "node_feat": np.random.randint(
                    size=(nodes_per_graph, num_node_feats),
                    low=np.zeros_like(node_feature_dims),
                    high=node_feature_dims,
                    dtype=np.int32,
                ),
                "edge_index": np.stack(self.get_random_edge_idx(nodes_per_graph, edges_per_graph)).astype(np.int32),
                "edge_feat": np.random.randint(
                    size=(edges_per_graph, num_edge_feats),
                    low=np.zeros_like(edge_feature_dims),
                    high=edge_feature_dims,
                    dtype=np.int32,
                ),
                "ogb_conformer": np.random.rand(nodes_per_graph, 3),
            }
            for _ in range(total_num_graphs)
        ]
        self.smiles = ["CC(NCC[C@H]([C@@H]1CCC(=CC1)C)C)C" for _ in range(total_num_graphs)]
        # List of edge_index edge_feat node_feat num_nodes for each graph
        self.labels = np.random.uniform(low=0, high=2, size=(total_num_graphs,)).astype(labels_dtype)
        self.name = "generated"

        # Select 80 percent of graphs for training and 10 percent for validation and test
        train_split_end = (self.total_num_graphs // 10) * 8
        valid_split_end = train_split_end + 1 + (self.total_num_graphs // 10)
        self.split_dict = {
            "train": np.arange(0, train_split_end),
            "valid": np.arange(train_split_end + 1, valid_split_end),
            "test": np.arange(valid_split_end + 1, self.total_num_graphs),
        }

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def get_idx_split(self):
        return self.split_dict

    def load_smile_strings(self, with_labels=False):
        return self.smiles, self.labels

    @staticmethod
    def get_random_edge_idx(nodes_per_graph, edges_per_graph):
        """
        this function gets random edge_idx to look 'sort-of' like the real data
        """
        outputs = []
        # each graph has its own adjacency
        edge_idx = []
        for i in range(nodes_per_graph - 1):
            edge_idx.extend([[i, i + 1], [i + 1, i]])

        upper_right_coords = []
        for i in range(nodes_per_graph):
            for j in range(i + 2, nodes_per_graph):
                upper_right_coords.append((i, j))

        random_connections = np.random.permutation(upper_right_coords)

        for i, j in random_connections[: (edges_per_graph - len(edge_idx)) // 2]:
            edge_idx.extend([[i, j], [j, i]])

        # offset the adjacency matrices
        # NOTE: Removed the offset as this was complicating laplacian functionality
        # Want to reinstate this later
        outputs.append(np.array(edge_idx))
        return np.transpose(np.vstack(outputs))


class CustomOGBGraphData:
    def __init__(self, custom_graph_items):
        self.total_num_graphs = len(custom_graph_items)
        self.graphs = custom_graph_items

        for graph in self.graphs:
            graph["ogb_conformer"] = np.array(np.full([len(graph["node_feat"]), 3], np.nan), dtype=float)

        # List of edge_index edge_feat node_feat num_nodes for each graph
        self.labels = [np.nan] * self.total_num_graphs
        self.name = "custom"
        graph_idxs = np.arange(0, self.total_num_graphs)

        self.split_dict = {
            "train": graph_idxs,
            "valid": graph_idxs,
            "test-dev": graph_idxs,
            "test-challenge": graph_idxs,
        }

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


class CustomGraphData(OGBGraphData):
    def __init__(
        self,
        custom_graph_items,
        use_extended_features=False,
        use_conformers=False,
        trim_chemical_features=False,
        use_periods_and_groups=False,
        do_not_use_atomic_number=False,
        chemical_node_features=["atomic_num"],
        chemical_edge_features=["possible_bond_type"],
    ):
        super().__init__(
            use_extended_features=use_extended_features,
            use_conformers=use_conformers,
            trim_chemical_features=trim_chemical_features,
            use_periods_and_groups=use_periods_and_groups,
            do_not_use_atomic_number=do_not_use_atomic_number,
            chemical_node_features=chemical_node_features,
            chemical_edge_features=chemical_edge_features,
        )
        self.labels_dtype = np.float32
        self.tf_labels_dtype = tf.float32
        self.dataset = CustomOGBGraphData(custom_graph_items)
        self.test_split_name = "valid"
        self.task_score_mode = "min"
        self.total_num_graphs = len(self.dataset.graphs)
        self.total_num_graphs_per_split = self.get_num_graphs_per_split()
        self.stats = self.get_statistics()
        self.check_idx = {"train": 0, "valid": 0, "test-dev": 0, "test-challenge": 0}
