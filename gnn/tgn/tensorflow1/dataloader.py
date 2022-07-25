# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Includes excerpts from PyTorch Geometric, examples/tgn.py
#   Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>
#   Licensed under the MIT License
#
"""Load JODIE-Wikipedia for training the TGN."""

import copy
import functools
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import tensorflow.compat.v1 as tf
import torch
import torch_geometric


class Data:
    """Data loading, batching, negative sampling & last neighour loading.

    This class wraps PyTorch Geometric's JODIEDataset and LastNeighborLoader to
    provide these functions:

        - Implement negative sampling to match PyG/examples/tgn.py
        - Pad batches to fixed shapes
        - Count batches in each partition {train, val, test}
        - Provide a separate `tf.data.Dataset` for each partition

    Dataset: JODIE-Wikipedia

    See: "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks",
    https://arxiv.org/abs/1908.01207,
    "Fast Graph Representation Learning with PyTorch Geometric",
    https://arxiv.org/abs/1903.02428.
    """

    Batch = Dict[str, np.ndarray]

    def __init__(self, path: Path, dtype: np.dtype, batch_size: int, nodes_size: int,
                 edges_size: int):
        self.data = torch_geometric.datasets.JODIEDataset(path,
                                                          name="wikipedia")[0]
        self.batch_size = batch_size
        # Upper bounds to pad nodes & edges within a batch. 
        self.nodes_size = 1 + nodes_size
        self.edges_size = edges_size
        train, val, test = self.data.train_val_test_split(val_ratio=0.15,
                                                          test_ratio=0.15)
        self.partitions = dict(train=train, val=val, test=test)
        feature_size = self.data.msg.shape[-1]
        self.batch_spec = dict(
            # Map from idx -> (global) node ID
            node_ids=((self.nodes_size, ), np.int32, -1),
            # Batch of events
            # ..(src, pos_dst, neg_dst)
            batch_idx=((3, self.batch_size), np.int32, self.nodes_size - 1),
            batch_times=((self.batch_size, ), np.int32, 0),
            batch_features=((self.batch_size, feature_size), dtype, 0.0),
            # ..mask most recent (src, pos_dst)
            batch_most_recent=((2, self.batch_size), np.bool_, False),
            # Context events (from neighbour loader)
            edge_idx=((2, self.edges_size), np.int32, self.nodes_size - 1),
            edge_times=((self.edges_size, ), np.int32, 0),
            edge_features=((self.edges_size, feature_size), dtype, 0.0),
        )

        # Precompute the correct starting state of LastNeighborLoader for each partition
        self.neighbour_loaders = {}
        neighbour_loader = torch_geometric.nn.models.tgn.LastNeighborLoader(
            self.data.num_nodes, size=10)
        self.neighbour_loaders["train"] = copy.deepcopy(neighbour_loader)
        for batch in self.partitions["train"].seq_batches(self.batch_size):
            neighbour_loader.insert(batch.src, batch.dst)
        self.neighbour_loaders["val"] = copy.deepcopy(neighbour_loader)
        for batch in self.partitions["val"].seq_batches(self.batch_size):
            neighbour_loader.insert(batch.src, batch.dst)
        self.neighbour_loaders["test"] = copy.deepcopy(neighbour_loader)

        # Also precompute neg_samples, but only for validation & test.
        dst_min, dst_max = int(self.data.dst.min()), int(self.data.dst.max())
        self.neg_samples = {}
        for part in ["val", "test"]:
            torch.manual_seed(12345)
            self.neg_samples[part] = [
                torch.randint(dst_min,
                              dst_max + 1,
                              batch.src.shape,
                              dtype=torch.long)
                for batch in self.partitions[part].seq_batches(self.batch_size)
            ]

    def n_batches(self, partition: str) -> int:
        """The exact total (padded) batch count for this partition."""
        return int(
            np.ceil(self.partitions[partition].num_events / self.batch_size))

    @staticmethod
    def most_recent_indices(indices: np.ndarray) -> np.ndarray:
        """Create a mask for the most recent (rightmost) instance of each index."""
        return ~np.any(
            np.triu(indices[np.newaxis] == indices[:, np.newaxis], 1), 1)

    def unpadded_batches(self, partition: str) -> Iterable[Batch]:
        """Generate unpadded numpy batches (encapsulates PyTorch bits)."""
        neighbour_loader = copy.deepcopy(self.neighbour_loaders[partition])
        dst_min, dst_max = int(self.data.dst.min()), int(self.data.dst.max())
        node_id_to_idx = torch.empty(self.data.num_nodes, dtype=torch.long)
        expected_count = self.n_batches(partition)
        for batch_n, batch in enumerate(self.partitions[partition].seq_batches(
                self.batch_size)):
            assert batch_n < expected_count
            neg_dst = (torch.randint(
                dst_min, dst_max +
                1, batch.src.shape, dtype=torch.long) if partition == "train"
                       else self.neg_samples[partition][batch_n])
            node_ids, edges, edge_ids = neighbour_loader(
                torch.cat([batch.src, batch.dst, neg_dst]).unique())
            node_id_to_idx[node_ids] = torch.arange(node_ids.shape[0])
            batch_idx = torch.stack([
                node_id_to_idx[ids] for ids in [batch.src, batch.dst, neg_dst]
            ]).numpy()
            # Transpose first because in "most recent" we want axis=1 (sequence)
            # ordered first, then axis=0 (src/dest)
            batch_most_recent = (self.most_recent_indices(
                batch_idx[:2].T.flatten()).reshape(-1, 2).T)
            yield dict(
                node_ids=node_ids.numpy(),
                batch_idx=batch_idx,
                batch_times=batch.t.numpy(),
                batch_features=batch.msg.numpy(),
                batch_most_recent=batch_most_recent,
                edge_idx=edges.numpy(),
                edge_times=self.data.t[edge_ids].numpy(),
                edge_features=self.data.msg[edge_ids].numpy(),
            )
            neighbour_loader.insert(batch.src, batch.dst)
        assert batch_n == expected_count - 1

    def _pad_batch(self, batch: Batch) -> Batch:
        assert batch.keys() == self.batch_spec.keys()
        assert (batch["node_ids"].shape[0] <= self.nodes_size -
                1), "node_ids requires at least 1 padding element"
        out = {}
        for key, (shape, dtype, pad_value) in self.batch_spec.items():
            value = batch[key]
            assert all(
                actual <= target for actual, target in zip(value.shape, shape)
            ), f"original shape {value.shape} larger than target {shape}"
            padding = [[0, target - actual]
                       for actual, target in zip(value.shape, shape)]
            out[key] = np.pad(value.astype(dtype),
                              padding,
                              constant_values=pad_value)
        return out

    def batches(self, partition: str) -> Iterable[Batch]:
        """Generate padded numpy batches of the correct dtype & shape."""
        return map(self._pad_batch, self.unpadded_batches(partition))

    def dataset(self, partition: str) -> tf.data.Dataset:
        """A TensorFlow dataset of batches."""
        return tf.data.Dataset.from_generator(
            functools.partial(self.batches, partition=partition),
            {key: dtype
             for key, (_, dtype, _) in self.batch_spec.items()},
            {key: shape
             for key, (shape, _, _) in self.batch_spec.items()},
        )
