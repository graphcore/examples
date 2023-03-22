# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf

try:
    from tensorflow.python import ipu
except ImportError:
    pass

from absl import flags

import layers
from data_utils.data_generators import EDGE_FEATURE_DIMS, NODE_FEATURE_DIMS

flags.DEFINE_enum("dtype", "float16", ("float16", "float32"), "data dtype")
flags.DEFINE_integer("n_latent", 300, "number of latent units in the network")
flags.DEFINE_integer("n_hidden", 600, "dimensionality for the hidden MLP layers")
flags.DEFINE_integer("n_mlp_layers", 2, "total number of layers in the MLPs (including output)")
flags.DEFINE_integer("n_embedding_channels", 100, "how many channels to use for the input embeddings")
flags.DEFINE_integer("n_graph_layers", 5, "how many message-passing steps in the model")
flags.DEFINE_integer("replicas", 1, "The number of replicas to scale the model over.")
flags.DEFINE_enum("opt", "adam", ("SGD", "adam"), "which optimizer to use")
flags.DEFINE_float("nodes_dropout", 0.0, "dropout for nodes")
flags.DEFINE_float("edges_dropout", 0.0, "dropout for edges")
flags.DEFINE_float("globals_dropout", 0.0, "dropout for globals")
flags.DEFINE_boolean("use_edges", True, "use edges in GIN")
flags.DEFINE_enum(
    "model", "graph_isomorphism", ("graph_network", "interaction_network", "graph_isomorphism"), help="model to use"
)

FLAGS = flags.FLAGS


def get_default_mlp(activate_final, name=None):
    return layers.MLP(
        n_layers=FLAGS.n_mlp_layers,
        n_hidden=FLAGS.n_hidden,
        n_out=FLAGS.n_latent,
        activate_final=activate_final,
        name=name,
    )


def create_model():
    """
    creates the GNN model
    params:
    bs -- batch size
    dtype -- data type
    eval_mode: used for checking correctness of the model function using
      pretrained weights
    """
    # nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx = graph
    inputs_list = [
        # inputs are categorical features
        tf.keras.Input((FLAGS.n_nodes_per_pack, NODE_FEATURE_DIMS), dtype=tf.int32, batch_size=FLAGS.micro_batch_size),
        tf.keras.Input((FLAGS.n_edges_per_pack, EDGE_FEATURE_DIMS), dtype=tf.int32, batch_size=FLAGS.micro_batch_size),
    ]
    if FLAGS.gather_scatter == "dense":
        inputs_list.extend(
            [
                # receivers
                tf.keras.Input(
                    (FLAGS.n_edges_per_pack, FLAGS.n_nodes_per_pack), dtype=tf.int32, batch_size=FLAGS.micro_batch_size
                ),
                # senders
                tf.keras.Input(
                    (FLAGS.n_edges_per_pack, FLAGS.n_nodes_per_pack), dtype=tf.int32, batch_size=FLAGS.micro_batch_size
                ),
                # node graph idx
                tf.keras.Input(
                    (FLAGS.n_nodes_per_pack, FLAGS.n_graphs_per_pack), dtype=tf.int32, batch_size=FLAGS.micro_batch_size
                ),
            ]
        )
    else:
        inputs_list.extend(
            [
                # receivers
                tf.keras.Input((FLAGS.n_edges_per_pack,), dtype=tf.int32, batch_size=FLAGS.micro_batch_size),
                # senders
                tf.keras.Input((FLAGS.n_edges_per_pack,), dtype=tf.int32, batch_size=FLAGS.micro_batch_size),
                # node graph idx
                tf.keras.Input((FLAGS.n_nodes_per_pack,), dtype=tf.int32, batch_size=FLAGS.micro_batch_size),
            ]
        )

    inputs_list.extend(
        [
            # edge graph idx
            tf.keras.Input((FLAGS.n_edges_per_pack,), dtype=tf.int32, batch_size=FLAGS.micro_batch_size),
        ]
    )

    if FLAGS.model == "graph_network":
        x = layers.EncoderLayer(
            edge_model_fn=lambda: get_default_mlp(activate_final=False, name="edge_encoder"),
            node_model_fn=lambda: get_default_mlp(activate_final=False, name="node_encoder"),
        )(inputs_list)
        for i in range(FLAGS.n_graph_layers):
            x = layers.GraphNetworkLayer(
                edge_model_fn=lambda: get_default_mlp(activate_final=False, name="edge"),
                node_model_fn=lambda: get_default_mlp(activate_final=False, name="node"),
                global_model_fn=lambda: get_default_mlp(activate_final=False, name="global"),
                # eval mode -- load all the weights, including the redundant last global layer
                nodes_dropout=FLAGS.nodes_dropout,
                edges_dropout=FLAGS.edges_dropout,
                globals_dropout=FLAGS.globals_dropout,
            )(x)
        output_logits = layers.DecoderLayer(
            global_model_fn=lambda: layers.MLP(
                n_layers=3, n_hidden=FLAGS.n_hidden, n_out=1, activate_final=False, name="output_logits"
            )
        )(x)

        # dummy dim needed -- see
        # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
        output_prob = tf.nn.sigmoid(output_logits)
        return tf.keras.Model(inputs_list, output_prob)
    elif FLAGS.model == "interaction_network":
        x = layers.EncoderLayer(
            edge_model_fn=lambda: get_default_mlp(activate_final=False, name="edge_encoder"),
            node_model_fn=lambda: get_default_mlp(activate_final=False, name="node_encoder"),
        )(inputs_list)
        for i in range(FLAGS.n_graph_layers):
            x = layers.InteractionNetworkLayer(
                edge_model_fn=lambda: get_default_mlp(activate_final=False, name="edge"),
                node_model_fn=lambda: get_default_mlp(activate_final=False, name="node"),
                nodes_dropout=FLAGS.nodes_dropout,
                edges_dropout=FLAGS.edges_dropout,
            )(x)
        output_logits = layers.DecoderLayer(
            global_model_fn=lambda: layers.MLP(
                n_layers=3, n_hidden=FLAGS.n_hidden, n_out=1, activate_final=False, name="output_logits"
            )
        )(x)

        # dummy dim needed -- see
        # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
        output_prob = tf.nn.sigmoid(output_logits)
        return tf.keras.Model(inputs_list, output_prob)
    elif FLAGS.model == "graph_isomorphism":
        graph_tuple = layers.GinEncoderLayer()(inputs_list)
        for i in range(FLAGS.n_graph_layers):
            graph_tuple = layers.GraphIsomorphismLayer(
                # final layer before output decoder is NOT activated
                get_mlp=lambda: get_default_mlp(name="GIN_mlp", activate_final=i < FLAGS.n_graph_layers),
                use_edges=FLAGS.use_edges,
                # edge embedding dimensionality must match the input to the layer
                edge_dim=FLAGS.n_latent if i > 0 else FLAGS.n_embedding_channels,
                dropout=FLAGS.nodes_dropout,
            )(graph_tuple)
        output_prob = layers.GinDecoderLayer()(graph_tuple)
        output_prob = tf.nn.sigmoid(output_prob)
        return tf.keras.Model(inputs_list, output_prob)
