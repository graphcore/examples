# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import numpy as np

from data_utils.mag_240_utils import PreprocessMAG240Dataset


def load_ogb_lsc_mag_dataset(dataset_path, dataset_name, pca_features_path):
    """Load the dataset from Open Graph Benchmark Large Scale Challenge package."""
    if dataset_name == "ogbn-lsc-mag240":
        logging.info(f"Loading MAG240 dataset. This dataset is very large (originally ~ 200Gb)"
                     " so loading + preprocessing may take ~ 1.5 hours.")
        mag_240_dataset = PreprocessMAG240Dataset(
            dataset_path,
            pca_features_path=str(dataset_path) + str(pca_features_path)
        )
        return load_ogb_lsc_mag240_dataset(mag_240_dataset)
    else:
        raise NotImplementedError(f"{dataset_name} is not currently supported"
                                  " in this applciation.")


def load_ogb_lsc_mag240_dataset(mag_240_dataset):

    author_paper_edges = mag_240_dataset.edge_index('author', 'writes', 'paper').astype(np.int32).T  # (386022720, 2)
    paper_paper_edges = mag_240_dataset.edge_index('paper', 'cites', 'paper').astype(np.int32).T  # (1297748926, 2)
    author_institutiuon_edges = mag_240_dataset.edge_index('author', 'affiliated_with', 'institution').astype(np.int32).T  # (44592586, 2)
    num_nodes = {"author": mag_240_dataset.num_authors, "paper": mag_240_dataset.num_papers, "institution": mag_240_dataset.num_institutions}
    # {'author': 122383112, 'paper': 121751666, 'institution': 25721}
    edges = {("author", "affiliated_with", "institution"): author_institutiuon_edges,  # (44592586, 2)
             ("author", "writes", "paper"): author_paper_edges,  # (386022720, 2)
             ("paper", "cites", "paper"): paper_paper_edges,  # (1297748926, 2) --> [np.array([3, 1806011]), np.array([3, 5950316]), ...]
             }

    pca_features = mag_240_dataset.pca_feat
    features = {'paper': pca_features[:mag_240_dataset.num_papers],
                'author': pca_features[mag_240_dataset.num_papers: mag_240_dataset.num_papers + mag_240_dataset.num_authors],
                'institution': pca_features[mag_240_dataset.num_papers + mag_240_dataset.num_authors:],
                }

    labels = {'paper': np.nan_to_num(mag_240_dataset.paper_label[:, np.newaxis], nan=-1).astype(np.int32)}  # (121751666,) mem_map of nan and floating point integers for classes

    # Rename test-dev to be in keeping with style in rest of application
    dataset_splits = mag_240_dataset.get_idx_split()
    dataset_splits["validation"] = dataset_splits.pop("valid")
    dataset_splits["test"] = dataset_splits.pop("test-dev")
    dataset_splits = {key: {"paper": np.array(values, dtype=np.int32),
                            "author": np.array([], dtype=np.int32),
                            "institution": np.array([], dtype=np.int32)}
                      for key, values in dataset_splits.items()}
    dataset_splits["train"]["author"] = np.unique(author_paper_edges[:, 0])
    dataset_splits["train"]["institution"] = np.unique(author_institutiuon_edges[:, 1])
    return num_nodes, edges, features, labels, dataset_splits
