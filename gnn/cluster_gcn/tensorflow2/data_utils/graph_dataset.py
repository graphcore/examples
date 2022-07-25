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

import gzip
import logging
import inspect
import os
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from utilities.constants import MASKED_LABEL_VALUE

PICKLE_GZ_EXT = ".pickle.gz"


class GraphDataset:
    """Base class for graph datasets holding the data and transforms
    to apply to the dataset. This should not be used directly, instead
    use its child classes."""
    def __init__(self,
                 dataset_name,
                 total_num_nodes,
                 edges,
                 features,
                 labels,
                 dataset_splits,
                 task,
                 graph_type,
                 skip_train_feats_and_edges_allocation=False):
        self.total_num_nodes = total_num_nodes
        self.edges = edges
        self.features = features
        self.labels = labels
        self.dataset_splits = dataset_splits
        self.task = task
        self.graph_type = graph_type
        self.dataset_name = dataset_name
        self.skip_train_feats_and_edges_allocation = skip_train_feats_and_edges_allocation

        self.meta_info = None

    @property
    def num_labels(self):
        raise NotImplementedError("Property 'num_labels' must be implemented by"
                                  "children classes")

    @property
    def num_features(self):
        raise NotImplementedError("Property 'num_features' must be implemented by"
                                  "children classes")

    @property
    def num_nodes(self):
        raise NotImplementedError("Property 'num_nodes' must be implemented by"
                                  "children classes")

    @property
    def num_edges(self):
        raise NotImplementedError("Property 'num_edges' must be implemented by"
                                  "children classes")

    def labels_to_one_hot(self, dtype=np.float32):
        raise NotImplementedError(
            "`labels_to_one_hot` method must be implemented in child class.")

    def normalize_features(self):
        raise NotImplementedError(
            "`normalize_features` method must be implemented in child class.")

    def precalculate_first_layer(self):
        raise NotImplementedError(
            "`precalculate_first_layer` method must be implemented in child class.")

    def add_undirected_connections(self):
        raise NotImplementedError(
            "`add_undirected_connections` method must be implemented in child class.")

    def remove_self_connections(self):
        raise NotImplementedError(
            "`remove_self_connections` method must be implemented in child class.")

    def generate_adjacency_matrices(self, dtype):
        raise NotImplementedError(
            "`generate_adjacency_matrices` method must be implemented in child class.")

    def generate_masks(self):
        raise NotImplementedError(
            "`generate_masks` method must be implemented in child class.")

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"\tName: {self.dataset_name},\n"
            f"\tNum features: {self.num_features},\n"
            f"\tNum labels: {self.num_labels},\n"
            f"\tTotal num nodes: {self.total_num_nodes},\n"
            f"\tNum nodes: {self.num_nodes},\n"
            f"\tNum edges: {self.num_edges},\n"
            f"\tGraph type: {self.graph_type},\n"
            f"\tTask: {self.task})"
        )

    @staticmethod
    def normalize(normalize_data, normalize_by_nodes):
        """Normalizes all the nodes in normalize_data by the mean and
        standard deviation of the node ids in normalize_by_nodes. The
        normalization is along the feature axis.
        :param normalize_data: An array where the values are features
            and indexes are the node ids with that feature.
        :param normalize_by_nodes: An array of node ids which are used
            to gather the mean and standard deviation to normalize by.
        :return: The array normalized array."""
        data_to_normalize_by = normalize_data[normalize_by_nodes]
        scaler = StandardScaler()
        scaler.fit(data_to_normalize_by)
        return scaler.transform(normalize_data)

    @staticmethod
    def precalculate_first_layer_features(features, adjacency):
        """Precalculate the exact and expensive AX."""
        adjacency = adjacency.astype(features.dtype)
        first_layer_features = adjacency.dot(features)
        return np.hstack((first_layer_features, features))

    @staticmethod
    def construct_adjacency(edges,
                            num_data,
                            dtype):
        """Get the (sparse) adjacency matrix for the graph
        given by a number of nodes and an edge list. Whether the
        graph is directed or not should also be provided. Undirected
        graphs will have the connections between nodes doubled.
        :param edges: An edge list that contains a list of tuples,
            where each tuple represents and edge from a sending
            to a receiving node index.
        :param num_data: The number of nodes in the full adjacency
            matrix.
        :param dtype: The data type of the returned adjacency matrix.
        :returns: Adjacency as a CSR matrix.
        """
        unsupported_dtypes = [np.float16]
        assert dtype not in unsupported_dtypes, (
            "The adjacency matrix is constructed as a CSR matrix which"
            f" doesn't support dtypes {unsupported_dtypes}. Either"
            " construct the adjacency with a supported dtype or cast"
            " the adjacency later in the dataset pipeline.")
        return sp.csr_matrix(
            (
                np.ones((edges.shape[0]), dtype=dtype),
                (edges[:, 0], edges[:, 1])
            ),
            shape=(num_data, num_data))

    @staticmethod
    def remove_self_connections_from_adjacency(adjacency):
        """Remove any self connections in the adjacency matrix"""
        if adjacency.diagonal().sum() > 0:
            # Change to list of lists (lil) format to make diagonal
            # operation more efficient.
            adjacency = adjacency.tolil()
            adjacency.setdiag(0)
            # Change back to compressed sparse row (csr) format as
            # it was previous.
            adjacency = adjacency.tocsr()
            logging.warning(
                "The adjacency matrix contained self connections, which"
                " have automatically been removed. If this is not"
                " desired, set the kwarg `remove_self_connections` to"
                " False.")
        return adjacency

    @staticmethod
    def add_undirected_connections_to_adjacency(adjacency):
        """Adds the undirected connections to the adjacency."""
        adjacency += adjacency.transpose()
        # Clip values to a maximum of 1.
        adjacency = adjacency.minimum(1)
        return adjacency

    @staticmethod
    def convert_to_one_hot(labels, dtype=np.float32):
        """Converts the labels to one hot labels. If the label
        is masked, then the generated one hot will be masked."""
        labels = labels.flatten()
        num_labels = max(labels) + 1
        num_nodes = len(labels)
        one_hot_labels = np.zeros((num_nodes, num_labels), dtype=dtype)
        for node, label in enumerate(labels):
            if label == MASKED_LABEL_VALUE:
                one_hot_labels[node, :] = MASKED_LABEL_VALUE
            else:
                one_hot_labels[node, label] = 1
        return one_hot_labels

    @staticmethod
    def create_sample_mask(idx, size):
        """Create a mask of size `size` with ones in the locations
        defined by idx, and zeros everywhere else."""
        mask = np.zeros(size)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    @staticmethod
    def generate_train_edge_list(edges,
                                 total_num_nodes,
                                 validation_data,
                                 test_data):
        """Generates the training edge list from the full edge list,
        total number of nodes and the nodes in each dataset split."""
        is_train = np.ones(total_num_nodes, dtype=np.bool)
        if validation_data is not None:
            is_train[validation_data] = False
            edges = edges[~np.in1d(edges[:, 0], validation_data)]
            edges = edges[~np.in1d(edges[:, 1], validation_data)]
        if test_data is not None:
            is_train[test_data] = False
            edges = edges[~np.in1d(edges[:, 0], test_data)]
            edges = edges[~np.in1d(edges[:, 1], test_data)]
        return edges

    @staticmethod
    def generate_train_features(full_features, training_nodes):
        """Generates the training features from the full features,
        and training nodes. This will be the same size as the full
        features with any features not for training set to 0."""
        masked_features = np.zeros_like(full_features)
        masked_features[training_nodes] = full_features[training_nodes]

        return masked_features

    def save(self, file_path):
        """Saves the preprocessed dataset data to file."""
        logging.info(f"Saving processed dataset to {file_path}...")
        with gzip.open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        # Give user rw, group rw and all r permissions
        os.chmod(file_path, 0o664)

    def save_mag(self, base_directory):
        """Generic function save each attribute in the class."""
        logging.info("Saving the MAG 240 dataset in a serialised fashion.")
        if not os.path.isdir(base_directory):
            os.mkdir(base_directory)
            # Give user rwx, group rwx and all r permissions
            os.chmod(base_directory, 0o774)

        def save_attribute(attrib):
            if inspect.isclass(attrib):
                attrib_name = attrib.__name__
            else:
                attrib_name = str(attrib).lower()
            attrib_filename = f"{base_directory}{attrib_name}{PICKLE_GZ_EXT}"
            with gzip.open(attrib_filename, "wb") as f:
                pickle.dump(getattr(self, attrib), f, protocol=4)
            # Give user rw, group rw and all r permissions
            os.chmod(attrib_filename, 0o664)

        for attribute in [*self.__dict__]:
            save_attribute(attribute)

        # Svae meta info including the key word args for base class to load into
        base_args_spec = inspect.getargspec(inspect.getmro(type(self))[1].__init__)
        self.meta_info = {"attributes_to_load": [*self.__dict__],
                          "type": type(self),
                          "base_args_spec": base_args_spec}
        save_attribute("meta_info")

    @classmethod
    def load_preprocessed_dataset(cls, file_path):
        """Loads the preprocessed dataset from file."""
        f = gzip.open(str(file_path), 'rb')
        dataset = pickle.load(f)
        return dataset

    @classmethod
    def load_preprocessed_mag240_dataset(cls, base_directory):
        logging.info("Loading the MAG 240 dataset in a serialised fashion.")

        meta_info_filename = f"{base_directory}meta_info{PICKLE_GZ_EXT}"
        meta_info = cls.load_preprocessed_dataset(meta_info_filename)

        # Load the key args needed to init the class first
        init_vars = meta_info["base_args_spec"]
        # Take the args (ignoring cls)
        init_dict = dict()
        for var in init_vars.args[1:]:
            var_filename = f"{base_directory}{var}{PICKLE_GZ_EXT}"
            init_dict[var] = cls.load_preprocessed_dataset(var_filename)
        # Take the class type from meta_info and unpack the vars in as keywords
        dataset = meta_info["type"](**init_dict)

        addition_vars = list(
            set(meta_info["attributes_to_load"]).difference(init_vars.args[1:])
        )
        for var in addition_vars:
            var_filename = f"{base_directory}{var}{PICKLE_GZ_EXT}"
            setattr(
                dataset,
                var,
                cls.load_preprocessed_dataset(var_filename)
            )

        return dataset


class HomogeneousGraphDataset(GraphDataset):
    """Homogeneous graph dataset class holding the data and transforms
    to apply to the dataset. All node types and edge types are the same."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.total_num_nodes == len(self.features), (
            "There is a mismatch between the number of entries for"
            " node features and number of nodes. There should be one"
            " for each node.")

        assert self.total_num_nodes == len(self.labels), (
            "There is a mismatch between the number of entries for"
            " node labels and number of nodes. There should be one"
            " for each node.")
        if self.skip_train_feats_and_edges_allocation is False:
            self.features_train = self.generate_train_features(
                self.features, self.dataset_splits["train"]
            )

            self.edges_train = self.generate_train_edge_list(
                self.edges,
                self.total_num_nodes,
                self.dataset_splits["validation"],
                self.dataset_splits["test"]
            )
        else:
            self.features_train = self.features
            self.edges_train = self.edges

        self.adjacency_train = None
        self.adjacency_full = None
        self.mask_train = None
        self.mask_validation = None
        self.mask_test = None

    @property
    def num_labels(self):
        """Returns the number of labels in the dataset."""
        num_labels = self.labels.shape[1]
        if num_labels == 1:
            # Then we assume the labels are sparse categorical representation
            num_labels = int(max(self.labels.flatten()) + 1)
        return num_labels

    @property
    def num_features(self):
        """Returns the number of features in the dataset."""
        return self.features.shape[1]

    @property
    def num_nodes(self):
        """Returns the number of nodes in the dataset for each split."""
        return {key: len(val) for key, val in self.dataset_splits.items()}

    @property
    def num_edges(self):
        """Returns the number of edges in the dataset for each split."""
        return {
            "train": len(self.edges_train),
            "validation": len(self.edges),
            "test": len(self.edges)
        }

    def labels_to_one_hot(self, dtype=np.float32):
        """Converts the labels to one hot labels."""
        logging.info(f"Converting labels to one hot vectors in dataset with dtype {dtype}...")
        self.labels = self.convert_to_one_hot(self.labels, dtype=dtype)

    def normalize_features(self):
        """Normalizes all features based on the training feature."""
        logging.info(f"Normalizing the features in dataset...")
        self.features_train = self.normalize(self.features_train, self.dataset_splits["train"])
        self.features = self.normalize(self.features, self.dataset_splits["train"])

    def features_to_dtype(self, dtype):
        """Casts all features to dtype."""
        logging.info(f"Casting the features in dataset to dtype {dtype.__name__}...")
        self.features_train = self.features_train.astype(dtype)
        self.features = self.features.astype(dtype)
        return self

    def labels_to_dtype(self, dtype):
        """Casts all labels to dtype."""
        logging.info(f"Casting the labels in dataset to dtype {dtype.__name__}...")
        self.labels = self.labels.astype(dtype)
        return self

    def precalculate_first_layer(self):
        """Precalculates the first layer features and concatenates
        them onto the existing features."""
        logging.info(f"Precalculating the first layer features in dataset...")
        self.features_train = self.precalculate_first_layer_features(
            self.features, self.adjacency_train)
        self.features = self.precalculate_first_layer_features(
            self.features, self.adjacency_full)

    def remove_self_connections(self):
        """Removes self connections from objects adjacency matrices."""
        logging.info(f"Removing self connections in dataset...")
        self.adjacency_train = self.remove_self_connections_from_adjacency(
            self.adjacency_train)
        self.adjacency_full = self.remove_self_connections_from_adjacency(
            self.adjacency_full)

    def add_undirected_connections(self):
        """Adds undirected connections from objects adjacency matrices."""
        logging.info(f"Removing undirected connections in dataset...")
        self.adjacency_train = self.add_undirected_connections_to_adjacency(
            self.adjacency_train)
        self.adjacency_full = self.add_undirected_connections_to_adjacency(
            self.adjacency_full)

    def generate_adjacency_matrices(self, dtype):
        """Generates the objects adjacency matrices from its edge lists."""
        logging.info(f"Generating adjacency matrices in dataset with dtype {dtype}...")
        self.adjacency_train = self.construct_adjacency(
            self.edges_train, self.total_num_nodes, dtype)
        self.adjacency_full = self.construct_adjacency(
            self.edges, self.total_num_nodes, dtype)

    def generate_masks(self):
        """Generates the objects masks."""
        logging.info(f"Generating masks in dataset...")
        self.mask_train = self.create_sample_mask(
            self.dataset_splits["train"], self.total_num_nodes)
        self.mask_validation = self.create_sample_mask(
            self.dataset_splits["validation"], self.total_num_nodes)
        self.mask_test = self.create_sample_mask(
            self.dataset_splits["test"], self.total_num_nodes)


class HeterogeneousGraphDataset(GraphDataset):
    """Heterogeneous graph dataset class holding the data and transforms
    to apply to the dataset. The edges and nodes have types associated
    with them."""

    def __init__(
        self,
        node_types,
        node_types_missing_features,
        node_types_missing_labels,
        node_types_missing_dataset_splits,
        edge_types,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.node_types = node_types
        self.node_types_missing_features = node_types_missing_features
        self.node_types_missing_labels = node_types_missing_labels
        self.node_types_missing_dataset_splits = node_types_missing_dataset_splits
        self.edge_types = edge_types

        self.features_train = {
            node_type: self.generate_train_features(self.features[node_type],
                                                    self.dataset_splits["train"][node_type])
            for node_type in self.node_types_with_features
        }

        self.edges_train = dict()
        for edge_type in self.edge_types:
            self.edges_train[edge_type] = np.vstack(
                [self.generate_train_edge_list(
                    self.edges[edge_type],
                    self.num_nodes_for_edge_type[edge_type],
                    self.dataset_splits["validation"].get(node_type, None),
                    self.dataset_splits["test"].get(node_type, None))
                 for node_type in self.node_types])
        logging.info("Preliminary processing done")

    @property
    def num_nodes_for_edge_type(self):
        """Returns the number of nodes involved in edge types."""
        return {
            edge_type: sum(
                [num_nodes for num_nodes in self.total_num_nodes.values()
                 for node_type in self.node_types if node_type in edge_type])
            for edge_type in self.edge_types
        }

    @property
    def node_types_with_features(self):
        """Returns the node types which contain features."""
        return tuple(
            [node_type for node_type in self.node_types
             if node_type not in self.node_types_missing_features]
        )

    @property
    def node_types_with_labels(self):
        """Returns the node types which contain labels."""
        return tuple(
            [node_type for node_type in self.node_types
             if node_type not in self.node_types_missing_labels]
        )

    @property
    def node_types_with_dataset_splits(self):
        """Returns the node types which dataset splits."""
        return tuple(
            [node_type for node_type in self.node_types
             if node_type not in self.node_types_missing_dataset_splits]
        )

    @property
    def num_labels(self):
        """Returns the number of labels in the dataset."""
        return self.labels[self.node_types_with_labels[0]].shape[1]

    @property
    def num_features(self):
        """Returns the number of features in the dataset."""
        return self.features[self.node_types_with_labels[0]].shape[1]

    @property
    def num_nodes(self):
        """Returns the number of nodes in the dataset for each split."""
        return {
            key: {k: len(v) for k, v in val.items()}
            for key, val in self.dataset_splits.items()
        }

    @property
    def num_edges(self):
        """Returns the number of edges in the dataset for each split."""
        return {
            "train": {edge_type: len(self.edges_train[edge_type])
                      for edge_type in self.edge_types},
            "validation": {edge_type: len(self.edges[edge_type])
                           for edge_type in self.edge_types},
            "test": {edge_type: len(self.edges[edge_type])
                     for edge_type in self.edge_types}
        }

    def generate_missing_labels(self, dtype=np.int32):
        """Generates masked labels for the nodes in the dataset without
        any label attributed to them."""
        logging.info(f"Generating missing labels in dataset...")
        for node_type in self.node_types_missing_labels:
            logging.warning(
                f"The labels for node type {node_type} will be populated with"
                f" values ({MASKED_LABEL_VALUE}). This will mean it is masked in"
                " training/validation/test.")
            self.labels[node_type] = np.full(
                (self.total_num_nodes[node_type], self.num_labels),
                np.array(MASKED_LABEL_VALUE, dtype=dtype),
                dtype=dtype)

    @staticmethod
    def sort_edge_list_by_institution(edges, mapping_rule):
        edge_list = edges[mapping_rule["edge_list"]]
        return edge_list[edge_list[:, 1].argsort()]

    def generate_missing_features(self, feature_mapping=None, dtype=np.float32):
        """Generates missing features for the nodes in the dataset without
        any features attributed to them.
        :param feature_mapping: A dictionary of dictionaries with entries that
        provides a mapping between the feature type to use to average and the
        list that links features to average to the node. If None no additional
        features are computed
        :param dtype: The precision to generate the features in
        """
        if feature_mapping is not None:
            if not isinstance(feature_mapping, list):
                logging.warning("Expected `feature_mapping` to be of type `list`. "
                                "This may cause errors if the new features are required"
                                "to be calculated in a specific order.")
            logging.info(f"Generating missing features in dataset...")
            # The missing features are made from averaging their neighbours
            # e.g. author features = average of their paper features
            #      institution features = average of their author features
            for mapping in feature_mapping:
                logging.info(f"Doing feature mapping for {mapping}.")
                feat_name, mapping_rule = mapping
                # Uses the edge list to find mapping to features
                # Edge list: ('author', 'affiliated_with', 'institution')
                # Author node list example: [0, 1, 2, 2, 2, 2, 3, 4]
                # Institution node list example: [845, 996, 3197, 6133, 6744, 7157, 5189, 7625]
                if feat_name == "institution":
                    # Sort the ('author', 'affiliated_with', 'institution') edge list
                    # by institution node number increasing
                    sorted_edge_list = self.sort_edge_list_by_institution(self.edges, mapping_rule)
                    # Obtain the list of unique authors and their indices for the next
                    # step of np.split. The first index is skipped as it is 0.
                    unique_institution_list_indices = np.unique(
                        sorted_edge_list[:, 1],
                        return_index=True
                    )[1][1:]
                    id_mapping = np.split(sorted_edge_list[:, 0], unique_institution_list_indices)
                # Edge list: ('author', 'writes', 'paper')
                # Author node list example: [0, 0, 0, 0, 1, 1, 1, 2]
                # Paper node list example:
                #  [19703, 289285, 311768, 402711, 181505, 297095, 336569, 14217]
                else:
                    # Obtain the list of unique authors and their indices for the
                    # next step of np.split. The first index is skipped as it is 0.
                    unique_author_list_indices = np.unique(
                        self.edges[mapping_rule["edge_list"]][:, 0],
                        return_index=True
                    )[1][1:]
                    id_mapping = np.split(
                        self.edges[mapping_rule["edge_list"]][:, 1],
                        unique_author_list_indices
                    )
                # Take the average of the features from the above mapping
                logging.info("Start feature averaging")
                self.features[feat_name] = np.vstack(
                    [
                        np.mean(self.features[mapping_rule["feature"]][x], axis=0)
                        for x in id_mapping
                    ]).astype(dtype)
                # Update the nodes with missing features
                logging.info("Finished feature averaging")
                self.node_types_missing_features = tuple(
                    set(self.node_types_missing_features) - set(tuple(self.features))
                )
        # All remaining nodes are assigned blank features
        for node_type in self.node_types_missing_features:
            self.features[node_type] = np.full(
                (self.total_num_nodes[node_type], self.num_features),
                0,
                dtype=dtype)

    def generate_missing_dataset_splits(self, dtype=np.int32):
        """Generates missing entries in the dataset_split for the nodes
        in the dataset that are missing. The missing nodes will be added
        to the training split."""
        logging.info(f"Generating missing dataset split entries in dataset...")
        for dataset_split, split in self.dataset_splits.items():
            if dataset_split == "train":
                # Add all nodes that aren't being trained on into the
                # training set, the labels of these will be masked out.
                for node_type in self.node_types_missing_dataset_splits:
                    split[node_type] = np.arange(
                        self.total_num_nodes[node_type],
                        dtype=dtype)
            else:
                for node_type in self.node_types_missing_dataset_splits:
                    split[node_type] = np.array([], dtype=dtype)

    def labels_to_one_hot(self, dtype=np.float32):
        raise NotImplementedError(
            "`labels_to_one_hot` method has not been implemented for"
            " heterogeneous graphs.")

    def normalize_features(self):
        raise NotImplementedError(
            "`normalize_features` method has not been implemented for"
            " heterogeneous graphs.")

    def precalculate_first_layer(self):
        raise NotImplementedError(
            "`precalculate_first_layer` method has not been implemented for"
            " heterogeneous graphs.")

    def add_undirected_connections(self):
        raise NotImplementedError(
            "`add_undirected_connections` method has not been implemented for"
            " heterogeneous graphs.")

    def remove_self_connections(self):
        raise NotImplementedError(
            "`remove_self_connections` method has not been implemented for"
            " heterogeneous graphs.")

    def generate_adjacency_matrices(self, dtype):
        raise NotImplementedError(
            "`generate_adjacency_matrices` method has not been implemented for"
            " heterogeneous graphs.")

    def generate_masks(self):
        raise NotImplementedError(
            "`generate_masks` method has not been implemented for"
            " heterogeneous graphs.")

    def features_to_dtype(self, dtype):
        raise NotImplementedError(
            "`features_to_dtype` method has not been implemented for"
            " heterogeneous graphs.")

    def labels_to_dtype(self, dtype):
        raise NotImplementedError(
            "`labels_to_dtype` method has not been implemented for"
            " heterogeneous graphs.")

    def reindex_edges(self):
        """Reindexes the edges so that, instead of having zero indexed
        node ids for each node type, each node type is indexed from the
        number of nodes in the previous node type, based on the ordering
        in self.node_type. This means there would be no overlapping node
        ids remaining in the graph."""
        # Reindex the node ids in the edge lists and combine
        for edge_type_key, edge_type_val in self.edges.items():
            # Store the id to reindex to next
            reindex_vals_to = 0
            for node_type in self.node_types:
                # Loop through the edge type tuple (eg. ('author', 'writes', 'paper'))
                for node_type_from_edge_idx, node_type_from_edge in enumerate(edge_type_key):
                    # If the current node type is in the edge type
                    if node_type == node_type_from_edge:
                        # If the node type is in the first position of the edge type
                        # we want to reindex element zero in the edge list, otherwise
                        # its in the last position and we want to update that position
                        # in the edge list.
                        index = 0 if node_type_from_edge_idx == 0 else 1
                        # Reindex the edges for this node type
                        edge_type_val[:, index] += reindex_vals_to
                # Update the id to reindex to next based on the number of nodes in
                # the node type we have just seen
                reindex_vals_to += self.total_num_nodes[node_type]

    def reindex_dataset_splits(self):
        """Reindexes the dataset splits so that, instead of having zero
        indexed node ids for each node type, each node type is indexed
        from the number of nodes in the previous node type, based on the
        ordering in self.node_type. This means there would be no overlapping
        node ids remaining in the graph."""
        for nodes in self.dataset_splits.values():
            # Store the id to reindex to next
            reindex_vals_to = 0
            for node_type in self.node_types:
                # Reindex the node ids for this node type
                nodes[node_type] += reindex_vals_to
                # Update the id to reindex to next based on the number of nodes in
                # the node type we have just seen
                reindex_vals_to += self.total_num_nodes[node_type]

    def to_homogeneous(self):
        """Returns a homogeneous graph dataset by combining the node and
        edge types of the heterogeneous graph dataset. This method combines
        the node and edge types into a single type and, because the node IDs
        are zero-based for a given node type, reindexes the node IDs."""

        logging.info(f"Converting heterogeneous dataset to homogeneous...")

        # Combine the total number of nodes for each node type
        num_nodes_entire_graph = sum(
            [num_nodes for num_nodes in self.total_num_nodes.values()])

        # Reindex the node ids in the edge lists and combine
        self.reindex_edges()
        # Combine the reindexed edges
        edges_list = [self.edges[edge_type] for edge_type in self.edge_types]
        edges_entire_graph = np.vstack(edges_list)

        # Combine the features, stacking based on the ordering in node_type
        assert all([node_type in self.features.keys() for node_type in self.node_types]), (
            "To convert a heterogeneous graph to homogeneous, all"
            f" node types ({self.node_types}) must have a feature."
        )
        feature_list = [self.features[node_type] for node_type in self.node_types]
        features_entire_graph = np.vstack(feature_list)

        # Combine the labels, stacking based on the ordering in node_type
        assert all([node_type in self.labels.keys() for node_type in self.node_types]), (
            "To convert a heterogeneous graph to homogeneous, all"
            f" node types ({self.node_types}) must have a label."
        )
        labels_list = [self.labels[node_type] for node_type in self.node_types]
        labels_entire_graph = np.vstack(labels_list)

        # Combine the dataset splits, as these contain node ids they
        # need to be reindexed.
        assert all(
            [node_type in nodes_split.keys()
             for nodes_split in self.dataset_splits.values()
             for node_type in self.node_types]
        ), (
            "To convert a heterogeneous graph to homogeneous, all"
            f" node types ({self.node_types}) must have a dataset_splits"
            " entry."
        )
        # Reindex the node ids in the dataset splits and combine
        self.reindex_dataset_splits()
        dataset_splits_entire_graph = dict()
        for split, nodes in self.dataset_splits.items():
            nodes_list = [nodes[node_type] for node_type in self.node_types]
            # Stack the reindexed node types together for each split
            dataset_splits_entire_graph[split] = np.hstack(nodes_list)
        return HomogeneousGraphDataset(
            dataset_name=self.dataset_name,
            total_num_nodes=num_nodes_entire_graph,
            edges=edges_entire_graph,
            features=features_entire_graph,
            labels=labels_entire_graph,
            dataset_splits=dataset_splits_entire_graph,
            task=self.task,
            graph_type=self.graph_type,
            skip_train_feats_and_edges_allocation=True
        )
