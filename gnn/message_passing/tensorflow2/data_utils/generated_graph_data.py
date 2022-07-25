# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import tensorflow as tf
from absl import flags
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


FLAGS = flags.FLAGS


class GeneratedGraphData:
    """
    This class makes randomly generated graph data for benchmarking
    these graphs are all the same size (no padding/dummy graphs) but it is still indicative of
      real performance
    """

    def __init__(self,
                 micro_batch_size,
                 latent_size=256,
                 dtype=np.float16,
                 nodes_per_graph=24,
                 edges_per_graph=50,
                 n_graphs_per_pack=1,
                 n_node_feats=9,
                 n_edge_feats=3,
                 batches_per_epoch=2048):
        self.micro_batch_size = micro_batch_size
        self.latent_size = latent_size
        self.dtype = dtype
        self.nodes_per_graph = nodes_per_graph
        self.edges_per_graph = edges_per_graph
        self.n_node_feats = n_node_feats
        self.n_edge_feats = n_edge_feats
        self.n_graphs_per_pack = n_graphs_per_pack

        # This low number means it is quick to generate the data: note that the same example may be seen many times
        self.n_examples = max(8092, micro_batch_size)

        self.batches_per_epoch = batches_per_epoch
        self.n_graphs_per_epoch = self.batches_per_epoch * micro_batch_size * n_graphs_per_pack

        # these are lists with the number of categorical options for each feature
        atom_feature_dims = get_atom_feature_dims()
        bond_feature_dims = get_bond_feature_dims()

        self.label_dtype = np.int32

        np.random.seed(23)
        self.data = {
            'nodes': np.random.randint(size=(self.n_examples, self.nodes_per_graph * self.n_graphs_per_pack,
                                             len(atom_feature_dims)),
                                       low=np.zeros_like(atom_feature_dims),
                                       high=atom_feature_dims, dtype=np.int32),
            'edges': np.random.randint(size=(self.n_examples, self.edges_per_graph * self.n_graphs_per_pack,
                                             len(bond_feature_dims)),
                                       low=np.zeros_like(bond_feature_dims),
                                       high=bond_feature_dims, dtype=np.int32),
            'edge_idx': np.stack(
                [self.get_random_edge_idx() for _ in range(self.n_examples)]).astype(np.int32),
            # binary ground truth
            'ground_truth': np.random.uniform(low=0, high=2, size=(self.n_examples, self.n_graphs_per_pack)).astype(self.label_dtype)
        }

    def get_random_edge_idx(self):
        """
        this function gets random edge_idx to look 'sort-of' like the real data
        """
        outputs = []
        # each graph has its own adjacency
        for graph_idx in range(self.n_graphs_per_pack):
            edge_idx = [[0, 1], [self.nodes_per_graph - 1, self.nodes_per_graph - 2]]
            for i in range(1, self.nodes_per_graph - 1):
                edge_idx.extend([[i, i + 1], [i, i - 1]])

            upper_right_coords = []
            for i in range(self.nodes_per_graph):
                for j in range(i + 2, self.nodes_per_graph):
                    upper_right_coords.append((i, j))

            random_connections = np.random.permutation(upper_right_coords)

            for i, j in random_connections[:(self.edges_per_graph - len(edge_idx)) // 2]:
                edge_idx.extend([[i, j], [j, i]])

            # offset the adjacency matrices
            outputs.append(np.array(edge_idx) + graph_idx * self.nodes_per_graph)

        return np.vstack(outputs)

    def batch_to_outputs(self, batch):
        # defensive copy
        batch = batch.copy()
        nodes = batch.pop('nodes')
        edges = batch.pop('edges')
        edge_idx = batch.pop('edge_idx')
        receivers, senders = edge_idx[..., 0], edge_idx[..., 1]

        # each synthetic graph has the same number of nodes/edges as the others
        node_graph_idx = tf.repeat(tf.range(self.n_graphs_per_pack), self.nodes_per_graph)
        node_graph_idx = tf.tile(node_graph_idx[None, :], [self.micro_batch_size, 1])

        edge_graph_idx = tf.repeat(tf.range(self.n_graphs_per_pack), self.edges_per_graph)
        edge_graph_idx = tf.tile(edge_graph_idx[None, :], [self.micro_batch_size, 1])
        ground_truth = batch.pop('ground_truth')
        assert not batch, "all fields of the batch must be used"

        nodes = tf.reshape(nodes, [self.micro_batch_size, -1, self.n_node_feats])
        edges = tf.reshape(edges, [self.micro_batch_size, -1, self.n_edge_feats])

        if FLAGS.gather_scatter == 'dense':
            receivers = tf.one_hot(receivers, depth=self.nodes_per_graph * self.n_graphs_per_pack)
            senders = tf.one_hot(senders, depth=self.nodes_per_graph * self.n_graphs_per_pack)
            # note: this would have to be different for doing multiple graphs per pack...
            node_graph_idx = tf.one_hot(node_graph_idx, depth=self.n_graphs_per_pack)

        sample_weights = tf.cast(tf.where(ground_truth == -1, 0, 1), self.label_dtype)
        ground_truth = tf.where(ground_truth == -1, tf.cast(0, self.label_dtype), ground_truth)
        # must have a final dummy dimension to match the output of the sigmoid
        ground_truth = ground_truth[..., None]

        return (nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx), ground_truth, sample_weights

    def get_tf_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(self.data)
        ds = ds.take(self.n_examples).cache().repeat().shuffle(1000).batch(
            self.micro_batch_size, drop_remainder=True)
        ds = ds.map(lambda b: self.batch_to_outputs(b))
        bs = FLAGS.global_batch_size or self.micro_batch_size
        # prefetch ALL the needed data, to ensure it isn't a bottleneck
        ds = ds.prefetch(bs * self.batches_per_epoch * FLAGS.replicas * FLAGS.epochs)
        return ds

    def get_ground_truth_and_masks(self):
        ground_truths = self.data["ground_truth"]
        include_sample_mask = ground_truths != -1.
        return ground_truths, include_sample_mask
