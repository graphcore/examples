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

import numpy as np

from sklearn.preprocessing import StandardScaler

from data_utils.generated_dataset_loader import generate_mock_graph_data
from data_utils.graphsage_dataset_loader import load_graphsage_data
from data_utils.ogb_dataset_loader import load_ogb_dataset
from data_utils.utils import construct_adj, sample_mask
from utilities.constants import Task
from utilities.options import ALLOWED_DATASET_TYPE


class Dataset:

    def __init__(self,
                 dataset_path,
                 dataset_name,
                 precalculate_first_layer):
        logging.info(f"Loading raw dataset...")
        if dataset_name == "arxiv":
            self.raw_dataset_tuple = load_ogb_dataset("ogbn-arxiv", dataset_path)
            self.directed_graph = True
            self.task = Task.MULTI_CLASS_CLASSIFICATION
        elif dataset_name == "generated":
            self.raw_dataset_tuple = generate_mock_graph_data()
            self.directed_graph = False
            self.task = Task.MULTI_CLASS_CLASSIFICATION
        elif dataset_name == "ppi":
            self.raw_dataset_tuple = load_graphsage_data(dataset_path, dataset_name)
            self.directed_graph = False
            self.task = Task.BINARY_MULTI_LABEL_CLASSIFICATION
        elif dataset_name == "reddit":
            self.raw_dataset_tuple = load_graphsage_data(dataset_path, dataset_name)
            self.directed_graph = False
            self.task = Task.MULTI_CLASS_CLASSIFICATION
        else:
            raise ValueError(f"Unrecognised dataset type: `{dataset_name}`."
                             f" Choose one of {ALLOWED_DATASET_TYPE}")
        logging.info(f"Raw dataset loaded.")

        (self.adjacency_train,
         self.adjacency_full,
         self.features_train,
         self.features_test,
         self.labels_train,
         self.labels_validation,
         self.labels_test,
         self.mask_train,
         self.mask_validation,
         self.mask_test,
         self.train_data,
         self.validation_data,
         self.test_data,
         self.num_data) = self.preprocess_dataset(
             *self.raw_dataset_tuple,
             directed_graph=self.directed_graph,
             precalculate_first_layer=precalculate_first_layer
         )
        self.visible_data_validation = np.arange(self.num_data)

    @property
    def num_labels(self):
        return self.labels_train.shape[1]

    @property
    def num_features(self):
        return self.features_train.shape[1]

    @staticmethod
    def normalize(normalize_data, normalize_by_nodes):
        data_to_normalize_by = normalize_data[normalize_by_nodes]
        scaler = StandardScaler()
        scaler.fit(data_to_normalize_by)
        return scaler.transform(normalize_data)

    @staticmethod
    def precalculate_first_layer_features(features, adjacency):
        first_layer_features = adjacency.dot(features)
        return np.hstack((first_layer_features, features))

    @classmethod
    def preprocess_dataset(
        cls,
        num_data,
        edges,
        features,
        labels,
        train_data,
        val_data,
        test_data,
        directed_graph,
        precalculate_first_layer,
        normalize_features=True,
    ):
        logging.info(f"Preprocessing dataset...")

        is_train = np.ones((num_data), dtype=np.bool)
        is_train[val_data] = False
        is_train[test_data] = False
        train_data = np.array([n for n in range(num_data) if is_train[n]],
                              dtype=np.int32)

        train_edges = [
            (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
        ]
        edges = np.array(edges, dtype=np.int32)
        train_edges = np.array(train_edges, dtype=np.int32)
        train_adj = construct_adj(train_edges, num_data, directed_graph)
        full_adj = construct_adj(edges, num_data, directed_graph)

        if normalize_features:
            features = cls.normalize(features, train_data)

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_data, :] = labels[train_data, :]
        y_val[val_data, :] = labels[val_data, :]
        y_test[test_data, :] = labels[test_data, :]

        train_mask = sample_mask(train_data, labels.shape[0])
        val_mask = sample_mask(val_data, labels.shape[0])
        test_mask = sample_mask(test_data, labels.shape[0])

        if precalculate_first_layer:
            train_feats = cls.precalculate_first_layer_features(features, train_adj)
            test_feats = cls.precalculate_first_layer_features(features, full_adj)
        else:
            train_feature_mask = np.zeros(features.shape)
            train_feature_mask[train_data] = 1
            train_feats = features*train_feature_mask
            test_feats = features

        logging.info(f"Dataset preprocessed.")

        return (train_adj,
                full_adj,
                train_feats,
                test_feats,
                y_train,
                y_val,
                y_test,
                train_mask,
                val_mask,
                test_mask,
                train_data,
                val_data,
                test_data,
                num_data)
