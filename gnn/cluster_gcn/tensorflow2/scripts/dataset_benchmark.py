# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import json
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.ipu.dataset_benchmark import dataset_benchmark

from data_utils.clustering_utils import ClusterGraph
from data_utils.dataset_batch_generator import tf_dataset_generator
from data_utils.dataset_loader import load_dataset
from model.precision import Precision
from utilities.argparser import add_arguments, combine_config_file_with_args
from utilities.constants import GraphType
from utilities.options import Options
from utilities.utils import get_adjacency_dtype, get_adjacency_form, get_method_max


def estimate_ds_throughput(config):

    # Set precision policy for training
    precision = Precision(config.training.precision)
    tf.keras.mixed_precision.set_global_policy(precision.policy)

    # Set how the adjacency matrix is expressed,
    # namely dense tensor, sparse tensor, or tuple.
    adjacency_form_training = get_adjacency_form(
        config.training.device,
        config.training.use_sparse_representation)
    # Decide on the dtype of the adjacency matrix
    adjacency_dtype_training = get_adjacency_dtype(
        config.training.device,
        config.training.use_sparse_representation)

    method_max_edges = get_method_max(config.method_max_edges)
    method_max_nodes = get_method_max(config.method_max_nodes)

    # Load the dataset
    dataset = load_dataset(
        dataset_path=config.data_path,
        dataset_name=config.dataset_name,
        precalculate_first_layer=config.model.first_layer_precalculation,
        adjacency_dtype=adjacency_dtype_training,
        features_dtype=precision.features_precision.as_numpy_dtype,
        labels_dtype=precision.labels_precision.as_numpy_dtype,
        regenerate_cache=config.regenerate_dataset_cache,
    )

    training_clusters = ClusterGraph(
        adjacency=dataset.adjacency_train,
        num_clusters=config.training.num_clusters,
        visible_nodes=dataset.dataset_splits["train"],
        max_nodes_per_batch=config.training.max_nodes_per_batch,
        clusters_per_batch=config.training.clusters_per_batch,
        dataset_name=config.dataset_name + "-training",
        cache_dir=config.data_path,
        regenerate_cluster_cache=config.regenerate_clustering_cache,
        directed_graph=(dataset.graph_type == GraphType.DIRECTED),
        adjacency_form=adjacency_form_training,
        inter_cluster_ratio=config.inter_cluster_ratio,
        method_max_edges=method_max_edges,
        method_max_nodes=method_max_nodes,
        node_edge_imbalance_ratio=config.cluster_node_edge_imbalance_ratio,
    )
    training_clusters.cluster_graph()

    # Create dataset generators for training
    data_generator_training = tf_dataset_generator(
        adjacency=dataset.adjacency_train,
        clusters=training_clusters.clusters,
        features=dataset.features_train,
        labels=dataset.labels,
        mask=dataset.mask_train,
        num_clusters=training_clusters.num_clusters,
        clusters_per_batch=training_clusters.clusters_per_batch,
        max_nodes_per_batch=training_clusters.max_nodes_per_batch,
        max_edges_per_batch=training_clusters.max_edges_per_batch,
        adjacency_dtype=adjacency_dtype_training,
        adjacency_form=adjacency_form_training,
        seed=config.seed
    )

    results_tfdatatype = dataset_benchmark(data_generator_training, config.training.epochs, elements_per_epochs = int(config.training.num_clusters / config.training.clusters_per_batch), print_stats=False)
    results_dict = json.loads(results_tfdatatype.numpy()[0].decode('utf-8'))
    throughputs = [epoch['elements_per_second'] * training_clusters.max_nodes_per_batch for epoch in results_dict['epochs']]
    skip_epochs = min(2, config.training.epochs-1)
    throughputs = throughputs[skip_epochs:]
    mean_throughput = np.mean(throughputs)
    min_throughput = np.min(throughputs)
    max_throughput = np.max(throughputs)
    std_throughput = np.std(throughputs)
    return mean_throughput, min_throughput, max_throughput, std_throughput


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # Prevent doubling of TF logs.
    tf.get_logger().propagate = False

    parser = argparse.ArgumentParser(description="Dataset benchmark")
    args = add_arguments(parser).parse_args()
    config = combine_config_file_with_args(args, Options)

    # Set log level based on config
    logging.getLogger().setLevel(config.logging)

    mean_tput, min_tput, max_tput, std_tput = estimate_ds_throughput(config)

    print(f'Mean throughput = {mean_tput:.1f} nodes/sec')
    print(f'Min throughput = {min_tput:.1f} nodes/sec')
    print(f'Max throughput = {max_tput:.1f} nodes/sec')
    print(f'STD throughput = {std_tput:.1f} nodes/sec')
