# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf

try:
    from tensorflow.python import ipu
except ImportError:
    pass

import layers
from absl import flags
from data import NODE_FEATURE_DIMS, EDGE_FEATURE_DIMS

flags.DEFINE_enum('dtype', 'float16', ('float16', 'float32'), 'data dype')
flags.DEFINE_integer('n_latent', 300, "number of latent units in the network")
flags.DEFINE_integer('n_hidden', 600, "dimensionality for the hidden MLP layers")
flags.DEFINE_integer('n_mlp_layers', 2, "total number of layers in the MLPs (including output)")
flags.DEFINE_integer('n_embedding_channels', 100, "how many channels to use for the input embeddings")
flags.DEFINE_integer('n_graph_layers', 5, "how many message-passing steps in the model")
flags.DEFINE_integer('num_ipus', 1, "how many IPUs to use")
flags.DEFINE_enum('opt', 'adam', ('SGD', 'adam'), "which optimizer to use")
flags.DEFINE_float("nodes_dropout", 0.0, "dropout for nodes")
flags.DEFINE_float("edges_dropout", 0.0, "dropout for edges")
flags.DEFINE_float("globals_dropout", 0.0, "dropout for globals")
flags.DEFINE_boolean('use_edges', False, 'use edges in GIN')
flags.DEFINE_enum("model", "graph_isomorphism",
                  ("graph_network", "interaction_network", "graph_isomorphism"),
                  help='model to use')

FLAGS = flags.FLAGS


def get_default_mlp(activate_final, name=None):
    return layers.MLP(n_layers=FLAGS.n_mlp_layers, n_hidden=FLAGS.n_hidden, n_out=FLAGS.n_latent,
                      activate_final=activate_final, name=name)


def create_model(dtype=tf.float16, eval_mode=False):
    """
    creates the GNN model
    params:
    bs -- batch size
    dtype -- data type
    eval_mode: used for checking correctness of the model function using
      pretrained weights
    """
    # nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx = graph

    bs = FLAGS.batch_size
    inputs_list = [
        tf.keras.Input((bs * FLAGS.n_nodes, NODE_FEATURE_DIMS), dtype=dtype, batch_size=1),
        tf.keras.Input((bs * FLAGS.n_edges, EDGE_FEATURE_DIMS), dtype=dtype, batch_size=1),
        tf.keras.Input((bs * FLAGS.n_edges, 1), dtype=tf.int32, batch_size=1),
        tf.keras.Input((bs * FLAGS.n_edges, 1), dtype=tf.int32, batch_size=1),
        # node graph idx
        tf.keras.Input((bs * FLAGS.n_nodes), dtype=tf.int32, batch_size=1),
        # edge graph idx
        tf.keras.Input((bs * FLAGS.n_edges), dtype=tf.int32, batch_size=1),
    ]

    inputs_list_squeezed = [tf.squeeze(_input) for _input in inputs_list]

    if FLAGS.model == 'graph_network':
        x = layers.EncoderLayer(
            edge_model_fn=lambda: get_default_mlp(activate_final=False, name='edge_encoder'),
            node_model_fn=lambda: get_default_mlp(activate_final=False, name='node_encoder'),
        )(inputs_list_squeezed)
        for i in range(FLAGS.n_graph_layers):
            x = layers.GraphNetworkLayer(
                edge_model_fn=lambda: get_default_mlp(activate_final=False, name='edge'),
                node_model_fn=lambda: get_default_mlp(activate_final=False, name='node'),
                global_model_fn=lambda: get_default_mlp(activate_final=False, name='global'),
                # eval mode -- load all the weights, including the redundant last global layer
                nodes_dropout=FLAGS.nodes_dropout,
                edges_dropout=FLAGS.edges_dropout,
                globals_dropout=FLAGS.globals_dropout
            )(x)
        output_logits = layers.DecoderLayer(
            global_model_fn=lambda: layers.MLP(
                n_layers=3,
                n_hidden=FLAGS.n_hidden,
                n_out=1,
                activate_final=False,
                name='output_logits')
        )(x)
    elif FLAGS.model == 'interaction_network':
        x = layers.EncoderLayer(
            edge_model_fn=lambda: get_default_mlp(activate_final=False, name='edge_encoder'),
            node_model_fn=lambda: get_default_mlp(activate_final=False, name='node_encoder'),
        )(inputs_list_squeezed)
        for i in range(FLAGS.n_graph_layers):
            x = layers.InteractionNetworkLayer(
                edge_model_fn=lambda: get_default_mlp(activate_final=False, name='edge'),
                node_model_fn=lambda: get_default_mlp(activate_final=False, name='node'),
                nodes_dropout=FLAGS.nodes_dropout,
                edges_dropout=FLAGS.edges_dropout,
            )(x)
        output_logits = layers.DecoderLayer(
            global_model_fn=lambda: layers.MLP(
                n_layers=3,
                n_hidden=FLAGS.n_hidden,
                n_out=1,
                activate_final=False,
                name='output_logits')
        )(x)
    elif FLAGS.model == 'graph_isomorphism':
        graph_tuple = layers.GinEncoderLayer()(inputs_list_squeezed)
        for i in range(FLAGS.n_graph_layers):
            graph_tuple = layers.GraphIsomorphismLayer(
                # final layer before output decoder is NOT activated
                get_mlp=lambda: get_default_mlp(name='GIN_mlp', activate_final=i < FLAGS.n_graph_layers - 1),
                use_edges=FLAGS.use_edges,
                # edge embedding dimensionality must match the input to the layer
                edge_dim=FLAGS.n_latent if i > 0 else FLAGS.n_embedding_channels,
                dropout=FLAGS.nodes_dropout
            )(graph_tuple)
        output_prob = layers.GinDecoderLayer()(graph_tuple)
        return tf.keras.Model(inputs_list, output_prob)

    # dummy dim needed -- see
    # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
    output_prob = tf.reshape(tf.nn.sigmoid(output_logits), [-1, 1])
    model = tf.keras.Model(inputs_list, output_prob)
    return model
