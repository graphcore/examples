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

import numpy as np
import tensorflow as tf

from data_utils.utils import sample_mask
from utilities.constants import MASKED_LABEL_VALUE


def preprocess_multicluster(adj,
                            parts,
                            features,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size):
    """Generate the batch for multiple clusters."""
    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    np.random.shuffle(parts)
    for _, st in enumerate(range(0, num_clusters, block_size)):
        pt = parts[st]
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            pt = np.concatenate((pt, parts[pt_idx]), axis=0)
        features_batches.append(features[pt, :])
        y_train_batches.append(y_train[pt, :])
        support_now = adj[pt, :][:, pt]
        support_batches.append(support_now)
        total_nnz += support_now.count_nonzero()

        train_pt = []
        for new_idx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(new_idx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return (features_batches,
            support_batches,
            y_train_batches,
            train_mask_batches)


def batch_generator_fn(
    adjacency,
    partitions,
    features,
    labels,
    mask,
    num_clusters,
    clusters_per_batch,
    max_nodes_per_batch
):
    """Sample batches from a graph, based on clusters."""
    # For every epoch
    while True:
        (
            features_batches,
            adjacency_batches,
            labels_batches,
            mask_batches
        ) = preprocess_multicluster(
            adjacency,
            partitions,
            features,
            labels,
            mask,
            num_clusters,
            clusters_per_batch
        )

        for pid in range(len(features_batches)):
            features_batch = features_batches[pid]
            adjacency_batch = adjacency_batches[pid]
            labels_batch = labels_batches[pid]
            mask_batch = mask_batches[pid]
            mask_batch = ~mask_batch
            labels_batch[mask_batch] = MASKED_LABEL_VALUE

            num_nodes = len(features_batch)

            # Apply padding
            padding = (0, max_nodes_per_batch - num_nodes)
            features_batch = np.pad(features_batch, (padding, (0, 0)))
            labels_batch = np.pad(labels_batch, (padding, (0, 0)), constant_values=-1)
            adjacency_batch = np.pad(adjacency_batch.toarray(), (padding, padding))

            inputs = dict(adjacency=tf.cast(adjacency_batch, tf.bool), features=features_batch)
            yield inputs, labels_batch


def tf_dataset_generator(
    adjacency,
    partitions,
    features,
    labels,
    mask,
    num_clusters,
    clusters_per_batch,
    max_nodes_per_batch,
    seed=None
):
    """Create a tf.data.Dataset of batches."""
    if seed:
        np.random.seed(seed)

    dataset = tf.data.Dataset.from_generator(
        lambda: batch_generator_fn(
            adjacency,
            partitions,
            features,
            labels,
            mask,
            num_clusters,
            clusters_per_batch,
            max_nodes_per_batch
        ),
        output_types=(
            dict(adjacency=tf.bool,
                 features=tf.float32),
            tf.int32
        ),
        output_shapes=(
            dict(adjacency=(max_nodes_per_batch, max_nodes_per_batch),
                 features=(max_nodes_per_batch, features.shape[1])),
            (max_nodes_per_batch, labels.shape[1])
        )
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
