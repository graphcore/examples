# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Includes derived work from https://github.com/KiddoZhu/NBFNet-PyG
#   Copyright (c) 2021 MilaGraph
#   Licensed under the MIT License

import logging
from functools import reduce
import torch
import poptorch


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=0)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(0)).sum(dim=1)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(0)).sum(dim=1)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum())
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match


def negative_sampling(batch, graph, num_nodes, num_negative, strict=True):
    head_id, pos_tail_id, relation_id = batch.t()
    batch_size = head_id.shape[0]

    # strict negative sampling vs random negative sampling
    if strict:
        tail_mask = strict_negative_mask(graph, num_nodes, head_id, pos_tail_id, relation_id)
        neg_tail_candidate = tail_mask.nonzero()[:, 1]
        num_tail_candidate = tail_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(tail_mask), num_negative, device=batch.device)
        index = (rand * num_tail_candidate.unsqueeze(-1)).long()
        index = index + (num_tail_candidate.cumsum(0) - num_tail_candidate).unsqueeze(-1)
        neg_tail_id = neg_tail_candidate[index]
    else:
        neg_tail_id = torch.randint(num_nodes, (batch_size, num_negative))

    tail_id = torch.cat([pos_tail_id.unsqueeze(-1), neg_tail_id], dim=-1)

    return head_id, tail_id, relation_id


def all_negative(batch, num_nodes):
    head_id, pos_tail_id, relation_id = batch.t()
    batch_size = head_id.shape[0]
    all_tail_id = torch.arange(num_nodes).unsqueeze(0).expand(batch_size, -1)
    tail_id = (pos_tail_id.unsqueeze(-1) - all_tail_id) % num_nodes
    return head_id, tail_id, relation_id


def strict_negative_mask(graph, num_nodes, head_id, tail_id, relation_id):
    """Make sure that for a given (h, r) batch we will NOT sample true tails as random
    negatives"""
    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = graph[:, [0, 2]]
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([head_id, relation_id], -1)
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = graph[edge_id, 1]
    sample_id = torch.arange(len(num_t_truth)).repeat_interleave(num_t_truth)
    tail_mask = torch.ones(len(num_t_truth), num_nodes, dtype=torch.bool)
    # assign 0s to the mask with the found true tails
    tail_mask[sample_id, t_truth_index] = 0
    tail_mask.scatter_(1, tail_id.unsqueeze(-1), 0)

    return tail_mask


def create_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    return logger


def log_results(logger, log: dict, epoch: int, partition: str):
    logger.info(f"Epoch {epoch} {partition}")
    for k, v in log.items():
        logger.info(f"   {k}: {v:.4f}")


def recomputation_checkpoint(module: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Annotates the output of a module to be checkpointed instead of recomputed"""

    def recompute_outputs(module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            return poptorch.recomputationCheckpoint(outputs)
        if isinstance(outputs, tuple):
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)

    return module.register_forward_hook(recompute_outputs)


def batch_index_select(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """index_select with the first dimension of table and index being a batch
    dimension. Equivalent to
    ```table.gather(1, index.unsqueeze(-1).expand(-1, -1, table.shape[-1]))```
    but more efficient.

    :param table: Table from which to gather. Shape (batch, columns, dim)
    :param index: Indices to gather from table. Shape (batch, num_indices)
    :return: Tensor of shape (batch, num_indices, dim)
    """
    batch_size, num_nodes, dim = table.shape
    index_broadcast = index + torch.tensor(range(batch_size)).unsqueeze(-1).long() * num_nodes
    index_broadcast = index_broadcast.flatten()
    table = table.reshape((batch_size * num_nodes, dim))

    return table.index_select(0, index_broadcast).reshape(batch_size, -1, dim)
