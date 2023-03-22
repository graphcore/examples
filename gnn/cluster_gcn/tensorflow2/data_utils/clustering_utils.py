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

import logging
import math
import os
import time
from tqdm import tqdm
from pathlib import Path

import metis
import networkx as nx
import numpy as np

from utilities.constants import AdjacencyForm, MethodMaxNodesEdges
from utilities.constants import CLUSTERING_CACHE_EXT
from utilities.utils import decompose_sparse_adjacency


class ClusterGraph:
    """
    Class that helps clustering a dataset given a max nodes per batch
    or a number of total clusters to use. Only one of max_nodes_per_batch
    or num_clusters is accepted. If one is provided the other will
    be calculated based on the number of nodes in the dataset and
    the number of clusters to sample in a batch.
    """

    def __init__(
        self,
        adjacency,
        clusters_per_batch,
        visible_nodes,
        max_nodes_per_batch=None,
        num_clusters=None,
        dataset_name=None,
        cache_dir=None,
        directed_graph=False,
        inter_cluster_ratio=0.0,
        method_max_nodes=MethodMaxNodesEdges.UPPER_BOUND,
        method_max_edges=MethodMaxNodesEdges.UPPER_BOUND,
        adjacency_form=AdjacencyForm.DENSE,
        node_edge_imbalance_ratio=None,
        seed=1,
        regenerate_cluster_cache=True,
        save_clustering_cache=True,
    ):
        """
        Initialises the class
        :param adjacency: Adjacency matrix in compressed sparse row
            representation (CSR).
        :param clusters_per_batch: The number of clusters to include in
            a single batch of data.
        :param visible_nodes: The original indices of nodes visible during
            the current phase.
        :param max_nodes_per_batch: The maximum number of nodes to include
            in a batch. Either this or num_clusters can be specified. If
            this is left as None, it will be inferred from the specified
            num_clusters.
         :param num_clusters: The total number of clusters to cluster the
            dataset into. Either this or max_nodes_per_batch can be
            specified. If this is left as None, it will be inferred from
            the specified max_nodes_per_batch.
        :param dataset_name: Name of the dataset to be clustered.
        :param cache_dir: Directory where the file with result of clustering
            the dataset should be stored.
        :param directed_graph: Boolean flag to specify a directed graph or not.
        :param inter_cluster_ratio: Gives the amount of extra room to keep
            edges between sampled clusters. It is given as the ratio of the
            maximum number edges inside the sampled clusters.
        :param method_max_nodes: Method used to compute the maximum number
            of nodes per batch.
        :param method_max_edges: Method used to compute the maximum number
            of edges per batch. Only relevant for sparse tuple representation.
        :param node_edge_imbalance_ratio: Tuple of floats greater than 1.0,
            the first value representing the constraint of the requirement
            to balance the nodes per cluster, the second to balance the edges
            per cluster. The lower the value, the stricter this constraint.
        :param adjacency_form: Representation of the adjacency matrix, either
            dense (i.e., tensor), tf.SparseTensor, or tuple.
        :param seed: Seed for Metis random generator.
        :regenerate_cluster_cache: Bool to set regeneration of clustering cache or not.
        """

        if num_clusters is None and max_nodes_per_batch is None:
            raise ValueError(
                "One of num_clusters or max_nodes_per_batch" " must be set. Currently both are set to None."
            )
        if num_clusters and max_nodes_per_batch:
            raise ValueError("Only one of num_clusters or max_nodes_per_batch" " can be set. Currently both are set.")
        if num_clusters is not None:
            if num_clusters < 1:
                raise ValueError(
                    "The number of requested clusters must be"
                    f" greater than 1, {num_clusters} clusters"
                    " has been requested."
                )
            if num_clusters < clusters_per_batch:
                raise ValueError(
                    "Provided clusters per batch"
                    f" {clusters_per_batch} is greater than"
                    f" the requested total number of clusters"
                    f" {num_clusters}. Ensure it is less than"
                    " or equal to."
                )

        self.adjacency = adjacency
        self.clusters_per_batch = clusters_per_batch
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.regenerate_cluster_cache = regenerate_cluster_cache
        self.save_clustering_cache = save_clustering_cache
        self._max_nodes_per_batch = max_nodes_per_batch
        self._max_edges_per_batch = None
        self.directed_graph = directed_graph
        self.idx_nodes = visible_nodes
        self.num_nodes = len(self.idx_nodes)
        self.num_all_nodes = self.adjacency.shape[0]
        self.adjacency_form = adjacency_form
        self.inter_cluster_ratio = inter_cluster_ratio
        self.method_max_nodes = method_max_nodes
        self.method_max_edges = method_max_edges
        self.seed = seed
        self._clusters = None

        if node_edge_imbalance_ratio is not None:
            assert len(node_edge_imbalance_ratio) == 2 and all([x > 1.0 for x in node_edge_imbalance_ratio]), (
                "`node_edge_imbalance_ratio` must contain 2 floats greater" " than 1.0"
            )
        self.node_edge_imbalance_ratio = node_edge_imbalance_ratio

        if num_clusters is None:
            self.num_clusters = self.get_num_clusters(self.num_nodes, max_nodes_per_batch, clusters_per_batch)
            logging.info(
                f"Max nodes per batch ({max_nodes_per_batch})"
                f" provided, inferred {self.num_clusters} clusters"
                f" given the dataset size ({self.num_nodes} nodes)"
                " and clusters per batch"
                f" ({clusters_per_batch})."
            )
        else:
            self.num_clusters = num_clusters

    @staticmethod
    def get_num_clusters(num_nodes, max_nodes_per_batch, clusters_per_batch):
        """Returns the number of clusters required given the number of nodes
        in the dataset and the number of nodes per batch."""
        max_nodes_per_cluster = max_nodes_per_batch // clusters_per_batch
        return math.ceil(num_nodes / max_nodes_per_cluster)

    @property
    def max_nodes_per_batch(self):
        if self._max_nodes_per_batch is None:
            if self._clusters is None:
                raise ValueError("`cluster_graph` must be run before accessing" " max_nodes_per_batch.")

            num_nodes_per_cluster = [len(pt) for pt in self._clusters]

            max_nodes_per_batch = self.get_max(self.method_max_nodes, num_nodes_per_cluster, self.clusters_per_batch)

            max_nodes_per_batch = self.add_fake_node_for_sparse_tuple(max_nodes_per_batch, self.adjacency_form)
            self._max_nodes_per_batch = max_nodes_per_batch
            logging.info(
                f"Number of clusters ({self.num_clusters})"
                f" provided, inferred max nodes per batch"
                f" {self._max_nodes_per_batch} given the dataset size"
                f" ({self.num_nodes} nodes) and clusters per batch"
                f" ({self.clusters_per_batch})."
            )
        return self._max_nodes_per_batch

    @max_nodes_per_batch.setter
    def max_nodes_per_batch(self, value):
        self._max_nodes_per_batch = value

    @property
    def max_edges_per_batch(self):
        if self._max_edges_per_batch is None:
            if self._clusters is None:
                raise ValueError("`cluster_graph` must be run before accessing" " max_edges_per_batch.")
            logging.info("Counting the number of edges per cluster...")
            num_edges_per_cluster = [self.adjacency[pt, :][:, pt].sum() for pt in tqdm(self._clusters)]

            max_edges_per_batch = self.get_max(self.method_max_edges, num_edges_per_cluster, self.clusters_per_batch)

            if self.adjacency_form == AdjacencyForm.SPARSE_TUPLE:
                max_edges_per_batch += self.add_edges_for_sparse_tuple(
                    max_edges_per_batch, self.max_nodes_per_batch, self.inter_cluster_ratio
                )
            logging.info(
                f"Inferred max num edges per batch {max_edges_per_batch}, "
                f"given number of clusters ({self.num_clusters}) and "
                f"clusters per batch ({self.clusters_per_batch})."
                f" This includes room for inter-cluster and self-loop edges."
            )

            self._max_edges_per_batch = max_edges_per_batch
        return self._max_edges_per_batch

    @max_edges_per_batch.setter
    def max_edges_per_batch(self, value):
        self._max_edges_per_batch = value

    @property
    def clusters(self):
        if self._clusters is None:
            raise ValueError("`cluster_graph` must be run before accessing" " clusters.")
        return self._clusters

    @property
    def use_cluster_cache(self):
        return self.cache_dir and self.dataset_name

    @property
    def unique_identifier(self):
        return (
            f"{self.dataset_name}-{self.adjacency_form.name}-"
            f"{self.method_max_nodes.name}-{self.method_max_edges.name}-"
            f"{self.inter_cluster_ratio}-{self.node_edge_imbalance_ratio}-"
            f"{self.clusters_per_batch}"
        )

    @staticmethod
    def get_max(method_max, num_per_cluster, num_sampled_clusters):
        """
        Returns a value based of the num per cluster values depending
        on the method provided. For example, when getting the maximum
        number of edges to pad/truncate to, you could use the average
        of all the edges of all of the clusters.
        """
        if method_max == MethodMaxNodesEdges.UPPER_BOUND:
            num_per_cluster.sort()
            return int(sum(num_per_cluster[-num_sampled_clusters:]))
        elif method_max == MethodMaxNodesEdges.AVERAGE:
            return int(np.ceil(num_sampled_clusters * np.mean(num_per_cluster)))
        elif method_max == MethodMaxNodesEdges.AVERAGE_PLUS_STD:
            return int(np.ceil(num_sampled_clusters * (np.mean(num_per_cluster) + np.std(num_per_cluster))))
        else:
            raise ValueError("Unrecognised method_max_nodes/edges")

    @staticmethod
    def add_fake_node_for_sparse_tuple(max_num_nodes, adjacency_form):
        """
        When dealing with sparse tuples in the IPU, it requires a fixed
        number of edges. In order to pad the edge list, we pad with dummy
        edges that go form a fake node to itself. This function returns
        the number of nodes by considering this fake node if needed.
        """
        if adjacency_form == AdjacencyForm.SPARSE_TUPLE:
            return max_num_nodes + 1
        else:
            return max_num_nodes

    @staticmethod
    def add_edges_for_sparse_tuple(max_edges, max_nodes, inter_cluster_ratio):
        # Add room for inter-cluster edges.
        extra_edges = math.ceil(inter_cluster_ratio * max_edges)
        # Add self-edges.
        extra_edges += max_nodes
        return extra_edges

    def cluster_graph(self):
        """
        Return the graph clustering. If cache dir and dataset name were
        provided when creating the class, it first attempts to load the
        pre-computed clustering from the cache path. If no pre-computed
        clustering is found, then it clusters the graph and saves the
        result for future faster loading.
        """
        if self.use_cluster_cache and not self.regenerate_cluster_cache:
            if self.load():
                logging.info("Clustering loaded from cache successfully.")
                return
            logging.info("Unable to find full clustering cache.")

        logging.info(f"Clustering graph with name {self.dataset_name}...")
        self.compute_clustering()

        if self.use_cluster_cache and self.save_clustering_cache:
            self.save()
        return

    def compute_clustering(self):
        """Partition a graph using METIS into the given clusters."""

        start_time = time.time()

        if self.num_clusters > 1:
            adjacency_to_cluster = self.adjacency.copy()

            # METIS cannot cluster a directed graph so we first make
            # it undirected just for clustering.
            if self.directed_graph:
                adjacency_to_cluster += adjacency_to_cluster.transpose()

            edge_list_to_cluster = decompose_sparse_adjacency(
                adjacency_to_cluster[self.idx_nodes, :][:, self.idx_nodes].asformat("coo")
            )[0]
            num_edges = len(edge_list_to_cluster)

            graph_to_cluster = nx.Graph()

            if self.node_edge_imbalance_ratio:
                # Attempt to balance nodes and edges per cluster
                logging.info(
                    "Nodes to edges imbalance ratio is set to"
                    f" {self.node_edge_imbalance_ratio}. This will mean metis will"
                    " cluster based on two constraints, balancing the number"
                    " of nodes and number of edges in each cluster with"
                    " this tolerances for each of those constraints. The"
                    " optimal values for this will be dependent on the dataset."
                )
                # Create a new graph with the original nodes
                graph_to_cluster.add_nodes_from(range(self.num_nodes), node_weight=100, edge_weight=0)
                # Add new fake nodes corresponding to the existing edges, with
                # a different set of weights applied. These fake nodes are only
                # added for clustering purposes. This ensures we can
                # ask metis to constrain more on the original nodes than
                # the new edge nodes, or vice versa. We pick weights such that
                # the weights on the original nodes and fake nodes are far apart.
                graph_to_cluster.add_nodes_from(
                    range(self.num_nodes, self.num_nodes + num_edges), node_weight=0, edge_weight=100
                )
                # Ensure the weights in the graph are used
                graph_to_cluster.graph["node_weight_attr"] = ["node_weight", "edge_weight"]

                # For each of the new nodes, add an edge between the original
                # nodes and the new nodes.
                sender_edges = []
                receiver_edges = []
                for edge_idx, edge_tuple in enumerate(edge_list_to_cluster):
                    sender, receiver = edge_tuple
                    fake_node_id = edge_idx + self.num_nodes
                    sender_edges.append((sender, fake_node_id))
                    receiver_edges.append((receiver, fake_node_id))
                graph_to_cluster.add_edges_from(receiver_edges)
                graph_to_cluster.add_edges_from(sender_edges)
                # Define the balance of each of the node and edge constraints
                load_imbalance_tolerance = self.node_edge_imbalance_ratio
                # We observed that using the recursive method gives better balance of
                # nodes and edges than the direct k-way cuts method on the datasets.
                recursive = True
            else:
                # By default metis will attempt to balance the nodes per cluster
                logging.info(
                    "Nodes to edges balance ratio is set to None. This will"
                    " mean metis will cluster attempting to balance the"
                    " number of nodes in each cluster."
                )
                graph_to_cluster.add_nodes_from(range(self.num_nodes))
                graph_to_cluster.add_edges_from(edge_list_to_cluster)
                load_imbalance_tolerance = None
                recursive = False

            # Remove self edges so it is in a valid format for METIS
            graph_to_cluster.remove_edges_from(nx.selfloop_edges(graph_to_cluster))

            _, groups = metis.part_graph(
                graph_to_cluster, self.num_clusters, seed=self.seed, recursive=recursive, ubvec=load_imbalance_tolerance
            )
        else:
            groups = [0] * self.num_nodes

        parts = [[] for _ in range(self.num_clusters)]
        for nd_idx in range(self.num_nodes):
            gp_idx = groups[nd_idx]
            nd_orig_idx = self.idx_nodes[nd_idx]
            parts[gp_idx].append(nd_orig_idx)

        self._clusters = [np.array(pt, dtype=np.int32) for pt in parts]

        self.validate_clusters()

        logging.info(f"Clustering completed in {time.time() - start_time :.3f} seconds.")

    def validate_clusters(self):
        """Validates the results of the clustering."""
        clustered_nodes = np.array([])
        for cluster_idx, cluster in enumerate(self._clusters):
            clustered_nodes = np.concatenate((clustered_nodes, cluster))
        clustered_nodes = np.sort(clustered_nodes)

        np.testing.assert_equal(clustered_nodes, np.sort(self.idx_nodes))

    def get_cache_file_name(self, param_name):
        """
        Return the file name where the result of computing the clusters
        should be stored for the given param_name.
        """
        if self.num_clusters:
            init_param = f"num_clusters-{self.num_clusters}"
        elif self.max_nodes_per_batch:
            init_param = f"max_nodes_per_batch-{self.max_nodes_per_batch}"
        else:
            raise Exception("Either `max_nodes_per_batch` or " "`num_clusters` should be specified.")
        filename = f"{param_name}-{self.unique_identifier}-" f"{init_param}{CLUSTERING_CACHE_EXT}"
        return filename

    def save(self):
        """Save the results of clustering to file."""
        self.save_param_to_cache("clusters", self.clusters)
        self.save_param_to_cache("max_nodes_per_batch", self.max_nodes_per_batch)
        self.save_param_to_cache("max_edges_per_batch", self.max_edges_per_batch)

    def save_param_to_cache(self, param_name, variable):
        """Saves param with value variable to a numpy file."""
        file_name = self.get_cache_file_name(param_name)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        file_name = Path(self.cache_dir).absolute().joinpath(file_name)
        logging.info(f"Saving {param_name} in {file_name}...")
        with open(file_name, "wb") as f:
            np.save(f, variable)
        # Give user rw, group rw and all r permissions
        os.chmod(file_name, 0o664)

    def load(self):
        """Load clustering from a file."""
        self._clusters = self.load_param_from_cache("clusters")
        self._max_nodes_per_batch = self.load_param_from_cache("max_nodes_per_batch")
        self._max_edges_per_batch = self.load_param_from_cache("max_edges_per_batch")
        if (
            self._clusters is not None
            and self._max_nodes_per_batch is not None
            and self._max_edges_per_batch is not None
        ):
            self._max_nodes_per_batch = int(self._max_nodes_per_batch)
            self._max_edges_per_batch = int(self._max_edges_per_batch)
            logging.info(
                f"Loaded max_nodes_per_batch value {self._max_nodes_per_batch} "
                f"and max_edges_per_batch value {self._max_edges_per_batch} for "
                f"num clusters {self.num_clusters} and clusters per batch "
                f"{self.clusters_per_batch}."
            )
            return True
        else:
            return False

    def load_param_from_cache(self, param_name):
        """Loads param with param_name from a numpy file."""
        file_name = self.get_cache_file_name(param_name)
        cache_path = Path(self.cache_dir).absolute().joinpath(file_name)
        if cache_path.is_file():
            logging.info(
                f"Loading {param_name} from cache {cache_path}, if this isn't"
                " desired either remove this file or set"
                " --regenerate-clustering-cache to `True`."
            )
            with open(cache_path, "rb") as f:
                return np.load(f, allow_pickle=True)
        return None
