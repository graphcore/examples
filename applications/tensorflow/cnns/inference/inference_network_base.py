# Copyright 2020 Graphcore Ltd.
"""Base class for setting up pre-trained Tensorflow inference network."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union

import tensorflow as tf
from get_weights import get_weights
from tensorflow.core.framework import graph_pb2
from tensorflow.tools.graph_transforms import TransformGraph

# Add model module to path
models_path = Path(Path(__file__).absolute().parent.parent)
sys.path.append(str(models_path))
from models.optimize_for_infer import fold_batch_norms  # noqa


class InferenceNetwork(object):
    """Constructs the inference graph, loads trained weights, optimizes graph for inference."""

    def __init__(self, input_shape: Tuple[int, int, int],
                 num_outputs: int,
                 batch_size: int,
                 data_type: str,
                 config: Dict,
                 checkpoint_dir: Optional[str] = "checkpoints"):
        """Initialize.

        Args:
            input_shape: Input dims in HWC format.
            num_outputs: Output dims.
            batch_size: Batch size.
            config: Dict of model specific configs.
            checkpoint_dir: Path to checkpoint dir.
        """

        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.data_type = data_type

        try:
            self.graph_input = config['graph_input']
        except KeyError:
            self.graph_input = "image_input"

        self.network_name = config['network_name']
        self.image_input = tf.placeholder(dtype=self.data_type,
                                          shape=(self.batch_size,) + tuple(input_shape),
                                          name=self.graph_input)

        graph, output_node_names = self.build_graph(config)

        if isinstance(graph, tf.Graph):
            graph = self.restore_weights(graph, checkpoint_dir, self.network_name, output_node_names,
                                         dtype=self.data_type)

        self.optimized_graph = self.optimize_for_inference(graph, output_node_names, self.graph_input)

        self.graph_output = output_node_names[0] + ":0"

    @staticmethod
    def preprocess_method() -> Callable:
        raise NotImplementedError

    @staticmethod
    def decode_method() -> Callable:
        raise NotImplementedError

    # Every child class must over-ride this method.
    def build_graph(self, config: Dict) -> [Union[tf.Graph, tf.compat.v1.GraphDef], List]:
        """Build network that takes image input and generates classifier output.

        Args:
            config: Model specific configuration dict.

        Returns: Tensorflow graph and list of names of output tensors.
        """
        raise NotImplementedError

    @staticmethod
    def restore_weights(graph: tf.Graph, checkpoint_dir: str, network_name: str, output_node_names: List, dtype: str):
        # Restore weights
        if tf.train.latest_checkpoint(checkpoint_dir) is None:
            logging.info('Checkpoint dir %s not found, attempting to download pre-trained weights.',
                         Path(checkpoint_dir).as_posix())
            get_weights(Path(checkpoint_dir), network_name, dtype)

        if tf.train.latest_checkpoint(checkpoint_dir) is None:
            raise ValueError(
                "Weight download failed. Please re-try downloading the weights using the `get_weights.py` script.")

        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            logging.info(f'Successfully restored imagenet weights for {network_name} model.')
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                                                               tf.get_default_graph().as_graph_def(),
                                                                               output_node_names)
        return graph_def

    @staticmethod
    def optimize_for_inference(frozen_graph_def: tf.GraphDef, output_node_names: List,
                               graph_input: str) -> graph_pb2.GraphDef:
        """Optimize graph for inference.

        Args:
            frozen_graph_def: Frozen graph definition
            output_node_names: Names of outputs
            graph_input: Name of the image input to the graph.

        Returns: Optimized inference graph definition.
        """
        logging.info('Starting graph optimization.')
        # Remove identity ops in initializers to allow fusing batch norm with conv in the next line
        optimized_graph_def = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph_def)
        optimized_graph_def = fold_batch_norms(optimized_graph_def)
        transforms = ['remove_nodes(op=Identity, op=CheckNumerics)', 'strip_unused_nodes',
                      'fold_constants(ignore_errors=true)']

        optimized_graph_def = TransformGraph(
            optimized_graph_def,
            [f"{graph_input}:0"],
            output_node_names,
            transforms)

        logging.info('Completed graph optimization.')
        return optimized_graph_def
