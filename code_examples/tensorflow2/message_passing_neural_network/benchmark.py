# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf

import xpu
from model import create_model
from absl import flags, app

import numpy as np
from utils import ThroughputCallback, print_trainable_variables

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

flags.DEFINE_integer('epochs', 5, 'number of epochs to run for')
flags.DEFINE_integer('batch_size', 128, 'compute batch size')
flags.DEFINE_integer('total_batch_size', None, 'total batch size (gradients will accumulate)')
flags.DEFINE_integer('n_nodes', 24, 'nodes per graph for the synthetic data')
flags.DEFINE_integer('n_edges', 50, 'edges per graph for the synthetic data')
flags.DEFINE_integer('n_node_feats', 9, 'number of input features for the nodes')
flags.DEFINE_integer('n_edge_feats', 3, 'number of input features for the edges')
flags.DEFINE_float('lr', 1e-5, "learning rate")
flags.DEFINE_boolean('execution_profile', False, "doing an execution profile")
FLAGS = flags.FLAGS


class SyntheticGraphData:
    """
    this class makes synthetic graph data for benchmarking
    these graphs are all the same size (no padding/dummy graphs) but it is still indicative of
      real performance
    """

    def __init__(self, batch_size, latent_size=256,
                 dtype=np.float16, nodes_per_graph=24, edges_per_graph=50,
                 n_examples=4096 * 32):
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.dtype = dtype
        self.nodes_per_graph = nodes_per_graph
        self.edges_per_graph = edges_per_graph
        self.n_examples = n_examples

    def get_random_edge_idx(self):
        """
        this function gets random edge_idx to look 'sort-of' like the real data
        """
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

        return np.array(edge_idx)

    def batch_to_outputs(self, batch):
        # adjacency represented in a block-sparse fashion
        idx_offset = tf.constant([i * self.nodes_per_graph for i in range(self.batch_size)],
                                 dtype=tf.int32)
        batch['edge_idx'] += idx_offset[:, None, None]

        nodes = batch.pop('nodes')
        edges = batch.pop('edges')
        receivers, senders = tf.split(batch.pop('edge_idx'), 2, axis=-1)

        node_graph_idx = tf.repeat(tf.range(self.batch_size), self.nodes_per_graph)[None, :]
        edge_graph_idx = tf.repeat(tf.range(self.batch_size), self.edges_per_graph)[None, :]

        ground_truth = batch.pop('ground_truth')
        assert not batch, "all fields of the batch must be used"

        nodes = tf.reshape(nodes, [1, -1, 9])
        edges = tf.reshape(edges, [1, -1, 3])
        receivers = tf.reshape(receivers, [1, -1])
        senders = tf.reshape(senders, [1, -1])
        return (nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx), ground_truth

    @tf.autograph.experimental.do_not_convert
    def get_tf_dataset(self):
        atom_feature_dims = get_atom_feature_dims()
        bond_feature_dims = get_bond_feature_dims()

        np.random.seed(23)
        data = {
            'nodes': np.random.randint(size=(self.n_examples, self.nodes_per_graph, len(atom_feature_dims)),
                                       low=np.zeros_like(atom_feature_dims),
                                       high=atom_feature_dims, dtype=np.int32),
            'edges': np.random.randint(size=(self.n_examples, self.edges_per_graph, len(bond_feature_dims)),
                                       low=np.zeros_like(bond_feature_dims),
                                       high=bond_feature_dims, dtype=np.int32),
            'edge_idx': np.stack(
                [self.get_random_edge_idx() for _ in range(self.n_examples)]).astype(np.int32),
            # binary ground truth
            'ground_truth': np.random.uniform(low=0, high=2, size=(self.n_examples, 1)).astype(np.int32)
        }
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.take(self.n_examples).cache().shuffle(32).repeat().batch(self.batch_size, drop_remainder=True)
        ds = ds.map(lambda b: self.batch_to_outputs(b))
        ds = ds.prefetch(1024)

        return ds


def main(_):
    tf.keras.mixed_precision.set_global_policy("float16" if FLAGS.dtype == 'float16' else "float32")
    strategy = xpu.configure_and_get_strategy()
    steps_per_epoch = 4096 // FLAGS.batch_size if not FLAGS.execution_profile else 16

    with strategy.scope():
        model = create_model()
        print_trainable_variables(model)

        losses = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        if FLAGS.opt == 'SGD':
            opt = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr)
        elif FLAGS.opt == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
        else:
            raise NotImplementedError()

        model.compile(
            optimizer=opt,
            loss=losses,
            metrics=[],
            steps_per_execution=steps_per_epoch if not FLAGS.execution_profile else 4
        )
        if xpu.IS_IPU and FLAGS.total_batch_size is not None:
            accumulation_steps = FLAGS.total_batch_size // FLAGS.batch_size
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=accumulation_steps)
        else:
            accumulation_steps = 1

        callbacks = [ThroughputCallback(samples_per_epoch=steps_per_epoch * FLAGS.batch_size * accumulation_steps)]

        data_gen = SyntheticGraphData(FLAGS.batch_size, nodes_per_graph=FLAGS.n_nodes,
                                      edges_per_graph=FLAGS.n_edges,
                                      latent_size=FLAGS.n_latent).get_tf_dataset()

        model.fit(data_gen,
                  steps_per_epoch=steps_per_epoch,
                  epochs=FLAGS.epochs if not FLAGS.execution_profile else 1,
                  callbacks=callbacks)


if __name__ == '__main__':
    app.run(main)
