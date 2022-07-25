# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Longest-pack-first histogram-packing."""
import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union

import numpy as np


def pack_using_dlpfhp(data_list, max_edges_per_pack, max_nodes_per_pack, max_graphs_per_pack,
                      heuristic=np.multiply):
    """Dual Longest-pack-first histogram-packing algorithm."""
    assert len(data_list[0]) == 3, "make sure the data-list has three parts per entry"

    data_list = [(e * n, e, n, count) for e, n, count in data_list]
    data_list.sort(reverse=True)

    max_size = heuristic(max_edges_per_pack, max_nodes_per_pack)
    # Initialize main strategy data dictionary.
    # The key indicates how many tokens are left for full length.
    # The value is a list of tuples, consisting of counts and respective packs.
    # A pack is a (sorted) list of sequence length values that get concatenated.
    tmp_strategies_per_length = defaultdict(list)
    strategies_per_length = defaultdict(list)

    for size, edges_length, nodes_length, n_sequences_to_bin in data_list:
        # the keys represent how much space is left to achieve the full length
        offset = 0
        while n_sequences_to_bin > 0:
            keys = [key for key in tmp_strategies_per_length if key >= size + offset]
            offset = max_size + 1 if not keys else min(keys) - size

            if (size + offset) in tmp_strategies_per_length:
                # reversed so the 'pop' is easier to index
                for i in reversed(range(len(tmp_strategies_per_length[size + offset]))):
                    len_edges, len_nodes, n_sequences_to_pack = tmp_strategies_per_length[size + offset][i]
                    if ((edges_length + sum(len_edges)) <= max_edges_per_pack and
                            (nodes_length + sum(len_nodes)) <= max_nodes_per_pack and
                            len(len_edges) < max_graphs_per_pack):
                        tmp_strategies_per_length[size + offset].pop(i)
                        new_len_edges = len_edges + [edges_length]
                        new_len_nodes = len_nodes + [nodes_length]

                        new_size = heuristic(max_edges_per_pack - sum(new_len_edges),
                                             max_nodes_per_pack - sum(new_len_nodes))
                        new_count = min(n_sequences_to_pack, n_sequences_to_bin)
                        # adjust strategies
                        if n_sequences_to_pack > new_count:
                            tmp_strategies_per_length[size + offset].append(
                                (len_edges, len_nodes, n_sequences_to_pack - new_count))

                        # get rid of the key if the value is []
                        if not tmp_strategies_per_length[size + offset]:
                            tmp_strategies_per_length.pop(size + offset)

                        tmp_strategies_per_length[new_size].append((new_len_edges, new_len_nodes, new_count))
                        n_sequences_to_bin -= new_count
                        offset = 0
                        break
            offset += 1
            if offset + size > max_size:
                new_size = heuristic(max_edges_per_pack - edges_length, max_nodes_per_pack - nodes_length)
                if new_size == 0:
                    strategies_per_length[0].append(([edges_length], [nodes_length], n_sequences_to_bin))
                else:
                    tmp_strategies_per_length[new_size].append(([edges_length], [nodes_length], n_sequences_to_bin))
                break

    # merge all strategies
    for key in tmp_strategies_per_length:
        strategies_per_length[key].extend(tmp_strategies_per_length[key])

    # flatten strategies dictionary
    strategy_set, strategy_repeat_count = [], []
    for key in strategies_per_length:
        for len_edges, len_nodes, n_sequences_to_pack in strategies_per_length[key]:
            strategy_set.append((len_edges, len_nodes))
            strategy_repeat_count.append(n_sequences_to_pack)

    # calculating efficiency
    n_empty_edges, n_empty_nodes, n_empty_graphs = 0, 0, 0
    for count, [pack_edges, pack_nodes] in zip(strategy_repeat_count, strategy_set):
        n_empty_edges += count * (max_edges_per_pack - sum(pack_edges))
        n_empty_nodes += count * (max_nodes_per_pack - sum(pack_nodes))
        n_empty_graphs += count * (max_graphs_per_pack - len(pack_edges))

    packs = int(sum(strategy_repeat_count))
    token_efficiency = (
        100 - n_empty_edges / (max_edges_per_pack * packs) * 100,
        100 - n_empty_nodes / (max_nodes_per_pack * packs) * 100,
        100 - n_empty_graphs / (max_graphs_per_pack * packs) * 100,
    )

    logging.info(
        f"Token efficiency:\n"
        f"Nodes: {token_efficiency[0]:.2f}%\n"
        f"Edges: {token_efficiency[1]:.2f}%\n"
        f"Graphs: {token_efficiency[2]:.2f}%")

    return strategy_set, np.array(strategy_repeat_count), token_efficiency


@dataclass
class StrategyPlanner:
    n_edges: List[int]
    n_nodes: List[int]
    max_edges_per_pack: int
    max_nodes_per_pack: int
    max_graphs_per_pack: int
    randomize: bool = True
    packs_per_epoch: Union[None, int] = None

    def __post_init__(self):
        # recording which ids go with which shapes
        max_edges, max_nodes = max(self.n_edges), max(self.n_nodes)
        assert max_edges < self.max_edges_per_pack, f"you have {max_edges} edges in one graph, which will not fit in " \
                                                    f"{self.max_edges_per_pack} max_edges_per_pack"
        assert max_nodes < self.max_nodes_per_pack, f"you have {max_nodes} nodes in one graph, which will not fit in " \
                                                    f"{self.max_nodes_per_pack} max_edges_per_pack"

        shape_to_idx_orig = defaultdict(list)
        for idx, (n_edge, n_node) in enumerate(zip(self.n_edges, self.n_nodes)):
            # indexing by (edges, nodes);
            shape_to_idx_orig[(n_edge, n_node)].append(idx)

        self.shape_to_idx_orig = shape_to_idx_orig
        # data list
        data_list = [(e, n, len(shape_to_idx_orig[(e, n)])) for e, n in shape_to_idx_orig]
        self.strategy_set, self.strategy_repeat_count, self.efficiency = pack_using_dlpfhp(data_list,
                                                                                           self.max_edges_per_pack,
                                                                                           self.max_nodes_per_pack,
                                                                                           self.max_graphs_per_pack)
        self.packs_per_epoch = sum(self.strategy_repeat_count)

    def pack_indices_generator(self):
        # shuffle the ids so that the same packs are generated with different samples
        while True:
            shape_to_idx = copy.deepcopy(self.shape_to_idx_orig)
            if self.randomize:
                for shape in shape_to_idx:
                    np.random.shuffle(shape_to_idx[shape])

            pack_indices = []
            for pack_shapes, n_repeats in zip(self.strategy_set, self.strategy_repeat_count):
                # for each list of ids that could make up the correct-sized pack
                for repeat_idx in range(n_repeats):
                    tmp_idx_list = []
                    for edge_shape, node_shape in zip(*pack_shapes):
                        tmp_idx_list.append(shape_to_idx[(edge_shape, node_shape)].pop())
                    pack_indices.append(tmp_idx_list)

            yield pack_indices
