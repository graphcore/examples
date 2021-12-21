# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf

from absl import flags
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

NODE_FEATURE_DIMS = len(get_atom_feature_dims())
EDGE_FEATURE_DIMS = len(get_bond_feature_dims())
FLAGS = flags.FLAGS


def np_batch_generator(n_nodes, n_edges, batch_size, data_subset, epochs=1, shuffle=True):
    """
    generates the batches in numpy

    shape of batch is going to be:
    nodes:     [max_nodes, n_node_feat]
    edge_idx:  [max_edges, 2]
    edge_feat: [max_edges, n_edge_feat]
    labels:    [max_graphs, 1]

    :param n_nodes: the number of nodes per graph (on average)
    :param n_edges: the number of edges per graph (on average)
    :param batch_size: desired batch size (1 is reserved for the dummy graph)
    :param data_subset: train|val|test data
    :param epochs: how many times to go through the data
    :return:
    """

    # we stop accumulating to a batch when one of these numbers is exceeded
    max_nodes = batch_size * n_nodes
    max_edges = batch_size * n_edges
    max_graphs = batch_size

    def get_new_batch_dict():
        # initializes an empty batch dictionary
        empty_keys = ("edge_features", "edge_idx", "labels")
        _batch_dict = {k: [] for k in empty_keys}
        _batch_dict['labels'] = [-1]
        _batch_dict['node_graph_idx'] = [0]
        _batch_dict['edge_graph_idx'] = []
        _batch_dict['node_features'] = [np.zeros([1, data_subset[0][0]['node_feat'].shape[1]])]
        return _batch_dict

    batch_dict = get_new_batch_dict()

    # node zero will be a dummy node that belongs to dummy graph 0
    n_nodes, n_edges, n_graphs = 1, 0, 1
    for _ in range(epochs):
        if shuffle:
            data_subset = np.random.permutation(data_subset)
        for this_graph, this_label in data_subset:
            this_graph_n_nodes, this_graph_n_edges = this_graph['num_nodes'], len(this_graph['edge_index'][0])
            new_n_nodes = n_nodes + this_graph_n_nodes
            new_n_edges = n_edges + this_graph_n_edges
            new_n_graphs = n_graphs + 1
            # node zero is a dummy node, so we reserve a place for that
            if new_n_nodes > (max_nodes - 1) or new_n_edges > max_edges or new_n_graphs > max_graphs:
                yield pad_graph(batch_dict, max_graphs, max_nodes, max_edges)
                # initialize next batch dictionary
                batch_dict = get_new_batch_dict()
                batch_dict['node_graph_idx'].extend([1] * this_graph_n_nodes)
                batch_dict['node_features'].append(this_graph['node_feat'])
                batch_dict['edge_graph_idx'].extend([1] * this_graph_n_edges)
                batch_dict['edge_features'].append(this_graph['edge_feat'])
                batch_dict['edge_idx'].append(this_graph['edge_index'] + 1)
                batch_dict['labels'].append(this_label)

                # reinitialize
                n_nodes = this_graph_n_nodes + 1
                n_edges = this_graph_n_edges
                # this_graph and the dummy graph
                n_graphs = 2
            else:
                batch_dict['node_graph_idx'].extend([n_graphs] * this_graph_n_nodes)
                batch_dict['node_features'].append(this_graph['node_feat'])
                batch_dict['edge_graph_idx'].extend([n_graphs] * this_graph_n_edges)
                batch_dict['edge_features'].append(this_graph['edge_feat'])
                # offsetting the edge indices by the accumulated number of nodes
                batch_dict['edge_idx'].append(this_graph['edge_index'] + n_nodes)
                batch_dict['labels'].append(this_label)

                n_nodes = new_n_nodes
                n_edges = new_n_edges
                n_graphs = new_n_graphs

    yield pad_graph(batch_dict, max_graphs, max_nodes, max_edges)


def pad_graph(graph_dict, n_graphs_post_padding, n_nodes_post_padding, n_edges_post_padding):
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
    node_graph_idx = np.zeros(n_nodes_post_padding)
    node_graph_idx[:len(graph_dict['node_graph_idx'])] = graph_dict['node_graph_idx']
    graph_dict['node_graph_idx'] = node_graph_idx

    node_features = np.concatenate(graph_dict['node_features'])
    padded_node_features = np.zeros([n_nodes_post_padding, node_features.shape[1]],
                                    dtype=node_features.dtype)
    padded_node_features[:len(node_features), :] = node_features
    graph_dict['node_features'] = padded_node_features

    edge_graph_idx = np.zeros(n_edges_post_padding)
    edge_graph_idx[:len(graph_dict['edge_graph_idx'])] = graph_dict['edge_graph_idx']
    graph_dict['edge_graph_idx'] = edge_graph_idx

    edge_features = np.concatenate(graph_dict['edge_features'])
    padded_edge_features = np.zeros([n_edges_post_padding, edge_features.shape[1]],
                                    dtype=edge_features.dtype)
    padded_edge_features[:len(edge_features), :] = edge_features
    graph_dict['edge_features'] = padded_edge_features

    edge_idx_padding = np.zeros(shape=[2, n_edges_post_padding - len(edge_features)], dtype=np.int32)
    # transpose so shape is [n_edge, 2]
    graph_dict['edge_idx'] = np.concatenate(graph_dict['edge_idx'] + [edge_idx_padding], axis=1).T

    labels_array = -np.ones([n_graphs_post_padding], dtype=np.int32)
    labels_array[:len(graph_dict['labels'])] = graph_dict['labels']
    graph_dict['labels'] = labels_array
    return graph_dict


def batch_to_outputs(batch):
    nodes = batch.pop('node_features')
    edges = batch.pop('edge_features')
    receivers, senders = tf.split(batch.pop('edge_idx'), 2, axis=-1)

    node_graph_idx = batch.pop('node_graph_idx')
    edge_graph_idx = batch.pop('edge_graph_idx')

    ground_truth = tf.reshape(batch.pop('labels'), [-1, 1])
    assert not batch, "all fields of the batch must be used"

    nodes = tf.reshape(nodes, [1, -1, NODE_FEATURE_DIMS])
    edges = tf.reshape(edges, [1, -1, EDGE_FEATURE_DIMS])
    receivers = tf.reshape(receivers, [1, -1])
    senders = tf.reshape(senders, [1, -1])

    sample_weights = tf.where(ground_truth == -1, 0., 1.)
    # ground truth should not have the padding
    ground_truth = tf.where(ground_truth == -1, 0, ground_truth)
    return (nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx), ground_truth, sample_weights


def get_tf_dataset(bs, subset, shuffle=True):
    nodes_per_graph = FLAGS.n_nodes
    edges_per_graph = FLAGS.n_edges
    ds = tf.data.Dataset.from_generator(
        lambda: np_batch_generator(nodes_per_graph, edges_per_graph, bs, subset,
                                   epochs=FLAGS.epochs, shuffle=shuffle),
        output_signature=(
            {
                'node_graph_idx': tf.TensorSpec(shape=(bs * nodes_per_graph), dtype=tf.int32),
                'node_features': tf.TensorSpec(shape=(bs * nodes_per_graph, 9), dtype=tf.float32),
                'edge_graph_idx': tf.TensorSpec(shape=(bs * edges_per_graph), dtype=tf.int32),
                'edge_features': tf.TensorSpec(shape=(bs * edges_per_graph, 3), dtype=tf.float32),
                'edge_idx': tf.TensorSpec(shape=(bs * edges_per_graph, 2), dtype=tf.int32),
                'labels': tf.TensorSpec(shape=(bs,), dtype=tf.int32),
            })
    )
    # repeating silences some errors (but won't affect any results)
    ds = ds.batch(1, drop_remainder=True).repeat()
    ds = ds.map(batch_to_outputs)
    return ds
