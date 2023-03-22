# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import math
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from ogb.graphproppred import GraphPropPredDataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from data_utils import packing_strategy_finder

NODE_FEATURE_DIMS = len(get_atom_feature_dims())
EDGE_FEATURE_DIMS = len(get_bond_feature_dims())


@dataclass
class PackedBatchGenerator:
    """
    Generates batched representations of graph data for various datasets.

    Each 'pack' of a batch is guaranteed to only exchange information with other members of the same 'pack'. This
      allows the use of 'batched' gather/scatter implementations. In this case, the compiler is able to use this
      constraint in order to improve throughput.
    """

    n_packs_per_batch: int
    max_graphs_per_pack: int
    max_nodes_per_pack: int
    max_edges_per_pack: int
    n_epochs: int = 100
    randomize: bool = True
    data_root: str = "./datasets/"
    fold: str = "train"
    dataset_name: str = "ogbg-molhiv"

    def __post_init__(self):
        # initializes the dataset and calculates the packing assignments
        self.dataset = GraphPropPredDataset(name=self.dataset_name)

        self.n_graphs_per_epoch = len(self.dataset.get_idx_split()[self.fold])
        self.dataset = [self.dataset[fold_idx] for fold_idx in self.dataset.get_idx_split()[self.fold]]

        self.n_edges = []
        self.n_nodes = []
        for g, _ in self.dataset:
            self.n_edges.append(len(g["edge_feat"]))
            self.n_nodes.append(g["num_nodes"])

        self.planned_strategy = packing_strategy_finder.StrategyPlanner(
            n_edges=self.n_edges,
            n_nodes=self.n_nodes,
            max_edges_per_pack=self.max_edges_per_pack,
            # packing so as to leave one node spare (dummy node) in each pack —
            #   dummy edges will connect this dummy node to itself
            max_nodes_per_pack=self.max_nodes_per_pack - 1,
            # the dummy node will broadcast its values to a dummy graph
            max_graphs_per_pack=self.max_graphs_per_pack - 1,
            randomize=self.randomize,
        )
        self.pack_indices_generator = self.planned_strategy.pack_indices_generator()
        self.packs_per_epoch = self.planned_strategy.packs_per_epoch
        self.batches_per_epoch = math.ceil(self.packs_per_epoch / self.n_packs_per_batch)
        self.pack_indices = []
        self.n_batches = self.n_epochs * self.n_packs_per_batch

        self.label_dtype = tf.int32

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pack_indices and self.n_batches > 0:
            self.pack_indices = next(self.pack_indices_generator)
            self.n_batches -= 1

        if not self.pack_indices:
            raise StopIteration

        current_pack_indices = self.pack_indices.pop()
        batch_dictionary = self.get_packed_datum(current_pack_indices)

        return batch_dictionary

    def get_ground_truth_and_masks(self):
        assert not self.randomize, "getting the ground truth and masks can only be done without randomization"
        local_pack_indices = next(self.pack_indices_generator)

        # -1. will represent masking
        ground_truths = np.full([self.packs_per_epoch, self.max_graphs_per_pack], -1.0)
        # 'reversed' matches the implicit reversal caused by 'popping' in the __next__ method
        for batch_idx, pack in enumerate(reversed(local_pack_indices)):
            for gt_idx, graph_idx_in_dataset in enumerate(pack):
                ground_truths[batch_idx, gt_idx] = self.dataset[graph_idx_in_dataset][1]

        include_sample_mask = ground_truths != -1.0
        return ground_truths, include_sample_mask

    def get_empty_batch_dict(self):
        # the dummy node is the last of each pack
        dummy_node_idx = self.max_nodes_per_pack - 1
        # the dummy graph is the last of each pack
        dummy_graph_idx = self.max_graphs_per_pack - 1

        batch_dict = dict()
        batch_dict["edge_graph_idx"] = np.full([self.max_edges_per_pack], dummy_graph_idx).astype(np.int32)
        batch_dict["edge_features"] = np.zeros([self.max_edges_per_pack, EDGE_FEATURE_DIMS], dtype=np.int32)
        batch_dict["edge_idx"] = np.full([self.max_edges_per_pack, 2], dummy_node_idx).astype(np.int32)

        batch_dict["node_graph_idx"] = np.full([self.max_nodes_per_pack], dummy_graph_idx).astype(np.int32)
        batch_dict["node_features"] = np.zeros([self.max_nodes_per_pack, NODE_FEATURE_DIMS], dtype=np.int32)
        # this is used for masking: for a graph id that corresponds to '-1' label, we will not include its loss
        batch_dict["labels"] = -np.ones([self.max_graphs_per_pack])
        return batch_dict

    def get_packed_datum(self, pack):
        packed_datum = self.get_empty_batch_dict()
        # we will count the entries as we pack, to maintain the IDs properly
        graph_ctr, edges_ctr, nodes_ctr = 0, 0, 0
        for graph_idx in pack:
            graph, ground_truth_label = self.dataset[graph_idx]
            this_graph_n_edges = len(graph["edge_feat"])
            packed_datum["edge_graph_idx"][edges_ctr : edges_ctr + this_graph_n_edges] = graph_ctr
            packed_datum["edge_features"][edges_ctr : edges_ctr + this_graph_n_edges, :] = graph["edge_feat"]
            # offsetting the edge indices by the accumulated number of nodes
            packed_datum["edge_idx"][edges_ctr : edges_ctr + this_graph_n_edges, :] = graph["edge_index"].T + nodes_ctr

            packed_datum["node_graph_idx"][nodes_ctr : nodes_ctr + graph["num_nodes"]] = graph_ctr
            packed_datum["node_features"][nodes_ctr : nodes_ctr + graph["num_nodes"], :] = graph["node_feat"]

            packed_datum["labels"][graph_ctr] = ground_truth_label

            graph_ctr += 1
            edges_ctr += this_graph_n_edges
            nodes_ctr += graph["num_nodes"]

        return packed_datum

    def get_tf_dataset(self):
        n_edges = self.max_edges_per_pack
        n_nodes = self.max_nodes_per_pack
        n_graphs = self.max_graphs_per_pack

        ds = tf.data.Dataset.from_generator(
            self.__iter__,
            output_signature=(
                {
                    "node_graph_idx": tf.TensorSpec(shape=(n_nodes,), dtype=tf.int32),
                    "node_features": tf.TensorSpec(shape=(n_nodes, NODE_FEATURE_DIMS), dtype=tf.float32),
                    "edge_graph_idx": tf.TensorSpec(shape=(n_edges,), dtype=tf.int32),
                    "edge_features": tf.TensorSpec(shape=(n_edges, EDGE_FEATURE_DIMS), dtype=tf.float32),
                    "edge_idx": tf.TensorSpec(shape=(n_edges, 2), dtype=tf.int32),
                    "labels": tf.TensorSpec(shape=(n_graphs,), dtype=self.label_dtype),
                }
            ),
        )
        # repeating silences some errors (but won't affect any results)
        ds = ds.batch(self.n_packs_per_batch, drop_remainder=True).repeat()
        ds = ds.map(self.batch_to_outputs)
        return ds

    def batch_to_outputs(self, batch):
        batch = batch.copy()
        nodes = batch.pop("node_features")
        edges = batch.pop("edge_features")
        edge_idx = batch.pop("edge_idx")
        receivers, senders = edge_idx[..., 0], edge_idx[..., 1]

        node_graph_idx = batch.pop("node_graph_idx")
        edge_graph_idx = batch.pop("edge_graph_idx")

        ground_truth = batch.pop("labels")
        assert not batch, "all fields of the batch must be used"

        # this COULD be moved above to the packed batch dict
        sample_weights = tf.cast(tf.where(ground_truth == -1, 0, 1), self.label_dtype)
        ground_truth = tf.where(ground_truth == -1, tf.cast(0, self.label_dtype), ground_truth)
        # must have a final dummy dimension to match the output of the sigmoid
        ground_truth = ground_truth[..., None]

        return (nodes, edges, receivers, senders, node_graph_idx, edge_graph_idx), ground_truth, sample_weights


if __name__ == "__main__":
    pbg = PackedBatchGenerator(
        n_packs_per_batch=6,
        n_epochs=1,
        max_graphs_per_pack=16,
        max_nodes_per_pack=248,
        max_edges_per_pack=512,
    )
    print(pbg.n_graphs_per_epoch)
    ds = pbg.get_tf_dataset()
    data = [batch for batch in ds]
    print(data[-1][-1])
