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
import scipy.sparse as sp
import tensorflow as tf

from utilities.constants import AdjacencyForm, MASKED_LABEL_VALUE
from utilities.utils import decompose_sparse_adjacency


SELF_EDGE_DUMMY_VALUE = -1


def add_self_edges_with_dummy_values(adjacency):
    """
    Add self-edges to the sparse representation, as they will be needed when
    processing (e.g., adding regularisation) the adjacency matrix with the
    current sparse ops. The values will be set to zeros in the batch generation.
    """
    # Check adjacency has zero diagonal.
    np.testing.assert_equal(adjacency.diagonal(), np.zeros((adjacency.shape[0],)))
    # Add self-edges (diagonal elements) with dummy value.
    # These will be set to zero in the batch generator but doing this
    # operation ahead of time saves time in the batch generator.
    return adjacency + SELF_EDGE_DUMMY_VALUE * sp.eye(adjacency.shape[0])


def set_self_edge_dummy_values_to_zero(adjacency):
    """
    Changes the adjacency representation from CSR to a tuple and replaces any
    dummy value with zeros for the self-edges.
    """
    # Convert from CSR to tuple representation.
    indices, values, shape = decompose_sparse_adjacency(
        adjacency.asformat("coo"))
    # Remove dummy values but keeping the self-edges.
    values = set_self_edges_values_to_zero(values)
    return indices, values, shape


def set_self_edges_values_to_zero(data):
    """
    Once we have the sparse matrix represented as a tuple, we can remove
    the dummy value and set the value of the self-edges to zero.
    """
    data[np.where(data == SELF_EDGE_DUMMY_VALUE)] = 0
    return data


def pad_adjacency_tuple(adjacency,
                        adjacency_dtype,
                        max_edges,
                        max_nodes):
    """
    Converts adjacency to a tuple of indices, values and shape, and pad
    the indices and values to a fixed size. The indices are padded with
    dummy self-edges of a fake node. The amount of padding includes the
    gap between the maximum number of edges in any batch and the number
    of edges in the current batch, plus some extra room for inter cluster
    edges. The values are padded to zero.
    """
    # Pad the nodes of the sparse adjacency by changing its shape.
    adjacency.resize((max_nodes, max_nodes))

    # Add self-edges, as they will be needed when processing
    # (e.g., adding regularisation) the adjacency matrix.
    indices, values, _ = set_self_edge_dummy_values_to_zero(
        adjacency)

    # Ensure values is the correct dtype
    values = values.astype(adjacency_dtype)

    # Get the number of edges in the current batch and
    # clip or pad to a fixed number of edges as needed.
    num_edges_in_batch = indices.shape[0]
    if num_edges_in_batch > max_edges:
        keep = np.random.choice(
            np.arange(0, num_edges_in_batch),
            size=max_edges,
            replace=False
        )
        indices = indices[keep, :]
        values = values[keep]
    else:
        edge_padding = (0, max_edges - num_edges_in_batch)
        # Add the new edges for padding to a fake node.
        fake_node_id = max_nodes - 1
        # Pad the edge list with self connections on this fake node.
        indices = np.pad(
            indices,
            (edge_padding, (0, 0)),
            constant_values=fake_node_id)
        # Pad the value list with zeros for the corresponding padded edges.
        values = np.pad(values, edge_padding, constant_values=0)
    return indices, values


def tf_dataset_generator(
    adjacency,
    clusters,
    features,
    labels,
    mask,
    num_clusters,
    clusters_per_batch,
    max_nodes_per_batch,
    max_edges_per_batch,
    adjacency_dtype,
    adjacency_form,
    micro_batch_size=1,
    seed=None,
    deterministic=False,
    prefetch_depth=10,
    distributed_worker_count=1,
    distributed_worker_index=0
):
    """Create a tf.data.Dataset of batches."""

    # Create a list of cluster indices that are cheaper to shuffle
    # than the full clusters list.
    cluster_indices = np.arange(num_clusters)

    # Convert cluster list of numpy arrays to list of lists, so that
    # concatenation is cheaper within the batch generation.
    cluster_lists = [list(c) for c in clusters]

    # Define a closure so this function has access to cluster_lists
    def get_nodes_from_cluster_indices(selected_cluster_indices):
        nodes_in_batch_list = []
        for cluster_idx in selected_cluster_indices:
            nodes_in_batch_list += cluster_lists[cluster_idx]
        return np.array(nodes_in_batch_list)

    # Create mask and apply to labels
    # The mask will be regenerated in the loss function based
    # on the value the masked labels are set to here.
    labels = labels.copy()
    mask = ~tf.cast(mask, tf.bool)
    labels[mask, :] = MASKED_LABEL_VALUE

    # For sparse operations, the matrix shape must be static, so we only need
    # two inputs: edges and values.
    if adjacency_form in (AdjacencyForm.SPARSE_TENSOR, AdjacencyForm.SPARSE_TUPLE):
        adjacency_type = (tf.int32, adjacency_dtype)
    else:
        adjacency_type = adjacency_dtype

    # Do the expensive parts of the sparse tuple preprocessing
    # ahead of time.
    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        adjacency = add_self_edges_with_dummy_values(adjacency)
    else:
        # We can cast the adjacency here for the sparse tensor and
        # dense case. For the sparse tuple we require the self edges
        # to have a dummy value which can't be represented as bool so
        # we cast later.
        adjacency = adjacency.astype(adjacency_dtype)

    def fix_output_shape_adjacency_dense(adjacency_batch,
                                         features_batch,
                                         labels_batch):
        # We must feed the IPU with a fixed shape tensor,
        # which for the dense representation is achieved by
        # padding with dummy nodes.
        adjacency_batch.set_shape((max_nodes_per_batch, max_nodes_per_batch))
        return adjacency_batch, features_batch, labels_batch

    def fix_output_shape_adjacency_sparse_tensor(indices_batch,
                                                 values_batch,
                                                 features_batch,
                                                 labels_batch):
        # We only use SparseTensor on CPU so can get away with not
        # specifying a fixed shape.
        indices_batch.set_shape((None, 2))
        values_batch.set_shape((None,))
        return indices_batch, values_batch, features_batch, labels_batch

    def fix_output_shape_adjacency_sparse_tuple(indices_batch,
                                                values_batch,
                                                features_batch,
                                                labels_batch):
        # We must feed the IPU with a tuple with fixed size, which is
        # achieved by padding when needed.
        indices_batch.set_shape((max_edges_per_batch, 2))
        values_batch.set_shape((max_edges_per_batch,))
        return indices_batch, values_batch, features_batch, labels_batch

    def fix_outputs_shape_feats_labels(x_batch, y_batch):
        x_batch["features_batch"].set_shape((max_nodes_per_batch, features.shape[1]))
        y_batch.set_shape((max_nodes_per_batch, labels.shape[1]))
        return x_batch, y_batch

    def process_adjacency_dense(nodes_in_batch,
                                features_batch,
                                labels_batch):
        adjacency_batch = adjacency[nodes_in_batch, :][:, nodes_in_batch]
        adjacency_batch = adjacency_batch.toarray()
        if max_nodes_per_batch > nodes_in_batch.size:
            # Convert to a dense matrix and pad with zero.
            node_padding = (0, max_nodes_per_batch - nodes_in_batch.size)
            adjacency_batch = np.pad(
                adjacency_batch, (node_padding, node_padding))
        return adjacency_batch, features_batch, labels_batch

    def process_adjacency_sparse_tensor(nodes_in_batch,
                                        features_batch,
                                        labels_batch):
        adjacency_batch = adjacency[nodes_in_batch, :][:, nodes_in_batch]
        # Pad the nodes of the sparse adjacency by changing its shape.
        adjacency_batch.resize((max_nodes_per_batch, max_nodes_per_batch))
        # Convert sparse matrix to a tuple that can be consumed by
        # tf.data.Dataset.from_generator.
        # We won't need the shape as it is a constant value.
        indices_batch, values_batch, _ = decompose_sparse_adjacency(
            adjacency_batch.asformat("coo"))
        return indices_batch, values_batch, features_batch, labels_batch

    def process_adjacency_sparse_tuple(nodes_in_batch,
                                       features_batch,
                                       labels_batch):
        """
        Converts adjacency to a tuple of indices, values and shape, and pad
        the indices and values to a fixed size. The indices are padded with
        dummy self-edges of a fake node. The amount of padding includes the
        gap between the maximum number of edges in any batch and the number
        of edges in the current batch, plus some extra room for inter cluster
        edges. The values are padded to zero.
        """
        adjacency_batch = adjacency[nodes_in_batch, :][:, nodes_in_batch]
        indices_batch, values_batch = pad_adjacency_tuple(
            adjacency_batch,
            adjacency_dtype,
            max_edges_per_batch,
            max_nodes_per_batch
        )
        return indices_batch, values_batch, features_batch, labels_batch

    def select_pad_features_and_labels(nodes_in_batch):
        num_nodes_in_batch = nodes_in_batch.size
        do_pad_nodes = max_nodes_per_batch > num_nodes_in_batch
        node_padding = None

        if not do_pad_nodes:
            keep = np.random.choice(
                np.arange(0, num_nodes_in_batch),
                size=max_nodes_per_batch,
                replace=False
            )
            nodes_in_batch = nodes_in_batch[keep]

        features_batch = features[nodes_in_batch, :]
        labels_batch = labels[nodes_in_batch, :]

        if do_pad_nodes:
            node_padding = (0, max_nodes_per_batch - nodes_in_batch.size)
            features_batch = np.pad(features_batch, (node_padding, (0, 0)))
            labels_batch = np.pad(
                labels_batch,
                (node_padding, (0, 0)),
                constant_values=MASKED_LABEL_VALUE
            )
        return nodes_in_batch, features_batch, labels_batch

    dataset = tf.data.Dataset.from_tensor_slices(cluster_indices)
    if distributed_worker_count > 1:
        dataset = dataset.shard(num_shards=distributed_worker_count,
                                index=distributed_worker_index)
    dataset = dataset.shuffle(num_clusters, seed=seed)
    dataset = dataset.repeat()
    dataset = dataset.batch(clusters_per_batch)

    dataset = dataset.map(
        lambda clusters_in_batch:
            tf.numpy_function(
                get_nodes_from_cluster_indices,
                [clusters_in_batch],
                clusters[0].dtype),
        num_parallel_calls=10,
        deterministic=deterministic)

    # Pad features and labels
    dataset = dataset.map(
        lambda nodes_in_batch:
            tf.numpy_function(
                select_pad_features_and_labels,
                [nodes_in_batch],
                (nodes_in_batch.dtype, features.dtype, labels.dtype)),
        num_parallel_calls=5,
        deterministic=deterministic)

    if adjacency_form == AdjacencyForm.DENSE:
        dataset = dataset.map(
            lambda nodes_in_batch, feats, labels:
                tf.numpy_function(
                    process_adjacency_dense,
                    [nodes_in_batch, feats, labels],
                    (adjacency_type, features.dtype, labels.dtype)),
            num_parallel_calls=5,
            deterministic=deterministic)
        dataset = dataset.map(fix_output_shape_adjacency_dense)
        dataset = dataset.map(
            lambda adj, feats, labels:
            (
                dict(
                    adjacency_batch=adj,
                    features_batch=feats
                ),
                labels
            )
        )
    elif adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        dataset = dataset.map(
            lambda nodes_in_batch, feats, labels:
                tf.numpy_function(
                    process_adjacency_sparse_tuple,
                    [nodes_in_batch, feats, labels],
                    (adjacency_type[0], adjacency_type[1], features.dtype, labels.dtype)),
            num_parallel_calls=5,
            deterministic=deterministic)
        dataset = dataset.map(fix_output_shape_adjacency_sparse_tuple)
        dataset = dataset.map(
            lambda adj_indices, adj_values, feats, labels:
            (
                dict(
                    adjacency_batch=(adj_indices, adj_values),
                    features_batch=feats
                ),
                labels
            )
        )
    elif adjacency_form == AdjacencyForm.SPARSE_TENSOR:
        dataset = dataset.map(
            lambda nodes_in_batch, feats, labels:
                tf.numpy_function(
                    process_adjacency_sparse_tensor,
                    [nodes_in_batch, feats, labels],
                    (adjacency_type[0], adjacency_type[1], features.dtype, labels.dtype)),
            num_parallel_calls=5,
            deterministic=deterministic)
        dataset = dataset.map(fix_output_shape_adjacency_sparse_tensor)
        dataset = dataset.map(
            lambda adj_indices, adj_values, feats, labels:
            (
                dict(
                    adjacency_batch=tf.sparse.SparseTensor(
                        indices=tf.cast(adj_indices, tf.int64),
                        values=adj_values,
                        dense_shape=tf.cast((max_nodes_per_batch, max_nodes_per_batch), tf.int64)
                    ),
                    features_batch=feats
                ),
                labels
            )
        )

    # Define shapes of features and labels
    dataset = dataset.map(fix_outputs_shape_feats_labels)

    if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
        assert micro_batch_size == 1, (
            f"A micro_batch_size of {micro_batch_size} has been provided,"
            " but only a micro_batch_size of 1 is currently supported.")
        dataset = dataset.batch(micro_batch_size,
                                drop_remainder=True,
                                num_parallel_calls=5,
                                deterministic=deterministic)

    dataset = dataset.prefetch(prefetch_depth)
    return dataset
