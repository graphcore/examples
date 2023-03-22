# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import pandas as pd
import plotly.express as px
import networkx as nx
import numpy as np
import wandb

from utilities.utils import decompose_sparse_adjacency


class ClusteringStatistics:
    """Class to calculate and plot a variety of statistics of the given
    clusters and adjacency."""

    def __init__(
        self, adjacency, clusters, num_clusters_per_batch, cluster_sample_size=500, combined_cluster_sample_size=100
    ):
        self.adjacency = adjacency
        self.clusters = clusters
        self.num_clusters_per_batch = num_clusters_per_batch
        self.cluster_sample_size = min(len(clusters), cluster_sample_size)
        self.combined_cluster_sample_size = combined_cluster_sample_size
        self.adjacency_coo = decompose_sparse_adjacency(adjacency.asformat("coo"))

        # Shuffle clusters to ensure random sampling
        np.random.shuffle(self.clusters)

    @staticmethod
    def get_graph_degrees(graph):
        """Returns the degrees for each of the nodes in the given graph."""
        return [val for (_, val) in graph.degree()]

    @staticmethod
    def build_nx_graph_from_edge_list(edge_list):
        """Given an edge list, returns a NetworkX graph object containing
        these edges."""
        graph = nx.Graph()
        graph.add_edges_from(edge_list.tolist())
        return graph

    @staticmethod
    def get_combined_clusters(clusters, clusters_per_batch, sample_size):
        """Combines the provided clusters into batches based on the number
        of clusters_per_batch. The sample size will limit the number of
        batches to sample."""
        combined_clusters = list()
        for _ in range(sample_size // (len(clusters) // clusters_per_batch) + 1):
            np.random.shuffle(clusters)
            for start in range(0, len(clusters), clusters_per_batch):
                end = min(start + clusters_per_batch, len(clusters))
                combined_clusters.append(np.concatenate(clusters[start:end]))
        return combined_clusters

    @staticmethod
    def get_sparsity_ratio(adjacency):
        """Returns the sparsity ratio of the given adjacency matrix"""
        num_non_zero_elements = adjacency.nnz
        num_elements = adjacency.shape[0] * adjacency.shape[1]
        if num_elements == 0:
            return 0
        return 1 - (num_non_zero_elements / num_elements)

    @staticmethod
    def get_sparsity_ratio_for_clusters(full_adjacency, clusters):
        """Returns the sparsity ratio of each of the provided clusters which
        are part of the full graph adjacency full_adjacency."""
        sparsity_ratios = []
        for cluster in clusters:
            cluster_adj = full_adjacency[cluster, :][:, cluster]
            sparsity_ratio = ClusteringStatistics.get_sparsity_ratio(cluster_adj)
            sparsity_ratios.append(sparsity_ratio)
        return sparsity_ratios

    @staticmethod
    def get_num_nodes_in_clusters(clusters, limit=None):
        """Returns the number of nodes in a cluster"""
        limit = limit if limit is not None else len(clusters)
        return [len(clusters[idx]) for idx in range(0, limit)]

    @staticmethod
    def get_cluster_degree(full_adjacency, cluster):
        """Returns the degree of a given cluster that is part of a full graph
        represented by full_adjacency."""
        cluster_edge_list = decompose_sparse_adjacency(full_adjacency[cluster, :][:, cluster].asformat("coo"))[0]
        cluster_graph = ClusteringStatistics.build_nx_graph_from_edge_list(cluster_edge_list)
        return ClusteringStatistics.get_graph_degrees(cluster_graph)

    @staticmethod
    def get_cluster_degrees(full_adjacency, clusters, limit=None):
        """Given a list of clusters, returns a list of degrees of each of
        the clusters. Each cluster is part of the graph represented by
        the adjacency matrix full_adjacency."""
        degrees = []
        limit = min(len(clusters), limit) if limit is not None else len(clusters)
        for idx in range(0, limit):
            cluster = clusters[idx]
            degrees.append(ClusteringStatistics.get_cluster_degree(full_adjacency, cluster))
        return degrees

    @staticmethod
    def plot_hist_to_wandb(data, name, data_name, **kwargs):
        """Plots a histogram to wandb."""
        df = pd.DataFrame(data, columns=[data_name])
        fig = px.histogram(df, x=data_name, **kwargs)
        wandb.log({f"{name}": fig})

    @staticmethod
    def plot_bar_graph_to_wandb(data, name, x_label, y_label):
        """Plots a bar graph to wandb."""
        table = wandb.Table(data=data, columns=[x_label, y_label])
        plot = wandb.plot.bar(table, x_label, y_label, title=name)
        wandb.log({name: plot})

    @staticmethod
    def plot_edge_summary_to_wandb(total_edges, clustered_edges, combined_cluster_edges):
        """Plots a summary of the number of edges to wandb."""
        values = [total_edges, clustered_edges, combined_cluster_edges]
        labels = ["total_edges", "clustered_edges", "combined_cluster_edges (batch)"]
        data = [[label, val] for (label, val) in zip(labels, values)]
        ClusteringStatistics.plot_bar_graph_to_wandb(data, "Edge comparison", "label", "num edges")

    @staticmethod
    def plot_min_mean_max_to_wandb(in_data, name, axis_label):
        """Plots min mean and max bar graph of the in_data to wandb."""
        values = [min(in_data), np.mean(in_data), max(in_data), np.std(in_data)]
        labels = ["min", "mean", "max", "std"]
        data = [[label, val] for (label, val) in zip(labels, values)]
        ClusteringStatistics.plot_bar_graph_to_wandb(data, name, "label", axis_label)

    def get_statistics(self, wandb):
        """Evaluates the statistics over the full graph, clusters and
        combined clusters, prints them to the terminal and plots them
        in wandb."""
        self.evaluate_full_graph()
        self.evaluate_clustered_graph()
        self.evaluate_combined_clustered_graph()
        self.print_statistics()
        if wandb is True:
            self.plot_statistics_to_wandb()
        else:
            logging.info(
                "Weights & Biases not enabled. To see plots enable" " wandb in the config or on the command line."
            )

    def evaluate_full_graph(self):
        """Evaluates the statistics of the full graph."""
        logging.info("Evaluating statistics for the full graph...")
        self.full_graph = self.build_nx_graph_from_edge_list(self.adjacency_coo[0])
        self.full_graph_degrees = self.get_graph_degrees(self.full_graph)
        self.total_degree = sum(self.full_graph_degrees)
        self.sparsity_full_graph = self.get_sparsity_ratio(self.adjacency)

    def evaluate_clustered_graph(self):
        """Evaluates the statistics of the clusters."""
        logging.info("Evaluating statistics for the clusters...")
        self.cluster_degrees = self.get_cluster_degrees(self.adjacency, self.clusters, limit=self.cluster_sample_size)
        self.edges_per_cluster = [sum(c) for c in self.cluster_degrees]
        self.nodes_per_cluster = self.get_num_nodes_in_clusters(self.clusters, limit=self.cluster_sample_size)
        self.sparsity_per_cluster = self.get_sparsity_ratio_for_clusters(self.adjacency, self.clusters)
        self.total_cluster_degrees = sum(self.edges_per_cluster) / (self.cluster_sample_size / len(self.clusters))

    def evaluate_combined_clustered_graph(self):
        """Evaluates the statistics of the clusters combined into batches."""
        logging.info("Evaluating statistics for the combined clusters (batches)...")
        combined_clusters = self.get_combined_clusters(
            self.clusters, self.num_clusters_per_batch, self.combined_cluster_sample_size
        )
        self.combined_cluster_degrees = self.get_cluster_degrees(
            self.adjacency, combined_clusters, limit=self.combined_cluster_sample_size
        )
        self.edges_per_combined_cluster = [sum(c) for c in self.combined_cluster_degrees]
        self.nodes_per_combined_cluster = self.get_num_nodes_in_clusters(
            combined_clusters, limit=self.combined_cluster_sample_size
        )
        self.sparsity_per_combined_cluster = self.get_sparsity_ratio_for_clusters(self.adjacency, combined_clusters)
        self.total_combined_cluster_degrees = sum(self.edges_per_combined_cluster) / (
            self.combined_cluster_sample_size / (len(self.clusters) // self.num_clusters_per_batch)
        )

    @staticmethod
    def formatted_min_mean_max(in_array):
        """Formatted print of the min mean and max of in_array."""
        return (
            f"min {min(in_array):.4f}, "
            f"mean {np.mean(in_array):.4f}, "
            f"max {max(in_array):.4f}, "
            f"std {np.std(in_array):.4f}"
        )

    def print_statistics(self):
        """Formatted print of the statistics."""
        cluster_percent = 100 * (self.total_cluster_degrees / self.total_degree)
        multi_cluster_percent = 100 * (self.total_combined_cluster_degrees / self.total_degree)
        logging.info(
            "Clustering statistics:\n"
            f"Full graph:\n"
            f"\t- Num nodes: {self.full_graph.number_of_nodes()}\n"
            f"\t- Num edges: {self.full_graph.number_of_edges()}\n"
            f"\t- Sparsity ratio: {self.sparsity_full_graph:.4f}\n"
            f"Clusters (sample size {self.cluster_sample_size}):\n"
            f"\t- Num nodes: {self.formatted_min_mean_max(self.nodes_per_cluster)}\n"
            f"\t- Num edges: {self.formatted_min_mean_max(self.edges_per_cluster)}\n"
            f"\t- Sparsity ratio: {self.formatted_min_mean_max(self.sparsity_per_cluster)}\n"
            f"\t- Percentage of remaining edges: {cluster_percent:.4f} %\n"
            f"Combined clusters (batches) (sample size {self.combined_cluster_sample_size}):\n"
            f"\t- Num nodes: {self.formatted_min_mean_max(self.nodes_per_combined_cluster)}\n"
            f"\t- Num edges: {self.formatted_min_mean_max(self.edges_per_combined_cluster)}\n"
            f"\t- Sparsity ratio: {self.formatted_min_mean_max(self.sparsity_per_combined_cluster)}\n"
            f"\t- Percentage of remaining edges: {multi_cluster_percent:.4f} %"
        )

    def plot_statistics_to_wandb(self):
        """Plots a variety of statistics to wandb."""
        logging.info("Plotting cluster statistics to wandb...")
        # Plot the node degree distribution for the entire graph
        # Get the 99% percentile to visualise plotting without edge cases
        perc = int(np.percentile(self.full_graph_degrees, 99))
        data = [x for x in self.full_graph_degrees if x < perc]
        self.plot_hist_to_wandb(data, "Node degrees of full graph", "node_degrees", nbins=perc, log_x=False)

        # Plot the node degree distribution for the clusters
        all_clusters = np.concatenate(self.cluster_degrees)
        perc = int(np.percentile(all_clusters, 99))
        data = [x for x in all_clusters if x < perc]
        self.plot_hist_to_wandb(data, "Node degrees of individual clusters", "node_degrees", nbins=perc, log_x=False)

        # Plot a summary of number of edges in graph and clusters
        self.plot_edge_summary_to_wandb(
            self.total_degree, self.total_cluster_degrees, self.total_combined_cluster_degrees
        )

        # Plot statistics for clusters
        self.plot_min_mean_max_to_wandb(self.edges_per_cluster, "Edges in clusters", "num_edges")
        self.plot_min_mean_max_to_wandb(self.nodes_per_cluster, "Nodes in clusters", "num_nodes")
        self.plot_min_mean_max_to_wandb(
            np.add(self.nodes_per_cluster, self.edges_per_cluster), "Nodes + edges in clusters", "num_nodes_and_edges"
        )
        self.plot_min_mean_max_to_wandb(self.sparsity_per_cluster, "Sparsity ratio in clusters", "Sparsity ratio")
        self.plot_bar_graph_to_wandb(
            [[idx, d] for idx, d in enumerate(np.sort(self.nodes_per_cluster))], "Nodes per cluster", "cluster", "nodes"
        )
        self.plot_bar_graph_to_wandb(
            [[idx, d] for idx, d in enumerate(np.sort(self.edges_per_cluster))], "Edges per cluster", "cluster", "edges"
        )

        # Plot statistics for combined clusters
        self.plot_min_mean_max_to_wandb(
            self.edges_per_combined_cluster, "Edges in combined cluster (batches)", "num_edges"
        )
        self.plot_min_mean_max_to_wandb(
            self.nodes_per_combined_cluster, "Nodes in combined cluster (batches)", "num_nodes"
        )
        self.plot_min_mean_max_to_wandb(
            np.add(self.nodes_per_combined_cluster, self.edges_per_combined_cluster),
            "Nodes + edges in combined cluster (batches)",
            "num_nodes_and_edges",
        )
        self.plot_min_mean_max_to_wandb(
            self.sparsity_per_combined_cluster, f"Sparsity ratio in combined cluster (batches)", "Sparsity ratio"
        )
        self.plot_bar_graph_to_wandb(
            [[idx, d] for idx, d in enumerate(np.sort(self.nodes_per_combined_cluster))],
            "Nodes per combined cluster (batches)",
            "combined cluster",
            "nodes",
        )
        self.plot_bar_graph_to_wandb(
            [[idx, d] for idx, d in enumerate(np.sort(self.edges_per_combined_cluster))],
            "Edges per combined cluster (batches)",
            "combined cluster",
            "edges",
        )
