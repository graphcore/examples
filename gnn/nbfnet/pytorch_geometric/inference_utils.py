# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Includes derived work from https://github.com/KiddoZhu/NBFNet-PyG
#   Copyright (c) 2021 MilaGraph
#   Licensed under the MIT License

import os
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from typing import Optional
import torch
from difflib import SequenceMatcher
from torch_scatter import scatter_add


class Prediction:
    def __init__(self, dataset, entities_path: str, raw_data_path: str):
        self.graph = torch.cat([dataset.edge_index.t(), dataset.edge_type.unsqueeze(1)], dim=1)
        self.num_nodes = dataset.num_nodes
        self.entity_vocab, self.relation_vocab = self.get_vocab(entities_path, raw_data_path)

    def get_vocab(self, entities_path, raw_data_path):
        entity_mapping = {}
        with open(entities_path, "r") as fin:
            for line in fin:
                k, v = line.strip().split("\t")
                entity_mapping[k] = v

        entity_vocab = []
        with open(os.path.join(raw_data_path, "entities.dict"), "r") as fin:
            for line in fin:
                id, e_token = line.strip().split("\t")
                entity_vocab.append(entity_mapping[e_token])

        relation_vocab = []
        with open(os.path.join(raw_data_path, "relations.dict"), "r") as fin:
            for line in fin:
                id, r_token = line.strip().split("\t")
                relation_vocab.append("%s (%s)" % (r_token.replace("_", " "), id))
                # relation_vocab.append("%s (%s)" % (r_token, id))

        return entity_vocab, relation_vocab

    def find_in_graph(self, head_id: str, relation_id: str):
        head_rel = self.graph[:, [0, 2]].numpy()
        tails = []
        for n, edge in enumerate(head_rel):
            if tuple(edge) == (head_id, relation_id):
                tails.append(int(self.graph[n, 1]))
        return tails, [self.entity_vocab[tail] for tail in tails]

    def print_triple(self, ind):
        rel_id = self.graph[ind, 2]
        annot = ""
        if rel_id >= 237:
            rel_id -= 237
            annot = "INVERSE "
        print(
            f"({self.entity_vocab[self.graph[ind,0]]}, {annot + self.relation_vocab[rel_id]}, {self.entity_vocab[self.graph[ind,1]]})"
        )

    def print_neighbours(self, head: str, max_print: int = 100):
        head_id, closest_head = match_string(head.lower(), self.entity_vocab)
        print(f"Closest match for query {closest_head}, ({head_id})")
        inds = (self.graph[:, 0] == head_id).nonzero()
        for ind in inds[: min(len(inds), max_print)]:
            self.print_triple(ind)

    def print_triples(
        self,
        head: Optional[str] = None,
        tail: Optional[str] = None,
        relation: Optional[str] = None,
        max_print: Optional[int] = 100,
        inverse: Optional[bool] = False,
    ):
        mask = torch.ones(self.graph.shape[0]).int()
        if head:
            head_id, closest_head = match_string(head.lower(), self.entity_vocab)
            print(f"Closest match for head query {closest_head}, ({head_id})")
            mask *= self.graph[:, 0] == head_id
        if tail:
            tail_id, closest_tail = match_string(tail.lower(), self.entity_vocab)
            print(f"Closest match for tail query {closest_tail}, ({tail_id})")
            mask *= self.graph[:, 1] == tail_id
        if relation:
            relation_id, closest_relation = match_string(relation.lower(), self.relation_vocab)
            print(f"Closest match for relation query {closest_relation}, ({relation_id})")
            if inverse:
                relation_id += len(self.relation_vocab)
            mask *= self.graph[:, 2] == relation_id
        inds = mask.nonzero()
        for ind in inds[: min(len(inds), max_print)]:
            print(ind)
            self.print_triple(ind)

    def inference(self, model, head: str, relation: str, top_k: int = 5, inverse: bool = False):
        model.eval()
        head_id, closest_head = match_string(head.lower(), self.entity_vocab)
        relation_id, closest_relation = match_string(relation.lower(), self.relation_vocab)
        print(f"Closest match for query {(closest_head, closest_relation)}, {(head_id, relation_id)}")
        if inverse:
            relation_id += len(self.relation_vocab)
        print(relation_id)
        tail_id = torch.arange(self.num_nodes).unsqueeze(0).long()
        prediction, paths, weights, _ = model(
            graph=self.graph + 1,
            num_nodes=self.num_nodes + 1,
            head_id=torch.tensor([head_id]) + 1,
            tail_id=tail_id + 1,
            relation_id=torch.tensor([relation_id]) + 1,
        )
        scores, inds = prediction[0].sort(descending=True)
        tails_in_graph, _ = self.find_in_graph(head_id, relation_id)
        return list(
            zip(
                [
                    self.entity_vocab[ind] + " " + str(ind.numpy()) + (" *" if ind in tails_in_graph else "")
                    for ind in inds[:top_k]
                ],
                scores[:top_k].numpy(),
            )
        )

    def path_importance(self, model, head_id, tail_id, relation_id):
        print(
            f"Finding paths ({self.entity_vocab[head_id], self.relation_vocab[relation_id], self.entity_vocab[tail_id]})"
        )
        model.eval()
        model.float()
        prediction, paths, weights, edge_weights = model(
            graph=self.graph,
            num_nodes=self.num_nodes,
            head_id=torch.tensor([head_id]),
            tail_id=torch.tensor([[tail_id]]),
            relation_id=torch.tensor([relation_id]),
            get_edge_importance=True,
        )
        edge_grads = torch.autograd.grad(prediction, edge_weights)
        distances, back_edges = beam_search_distance(self.graph, edge_grads, head_id, tail_id, self.num_nodes)
        paths, weights = topk_average_length(distances, back_edges, tail_id, k=4)

        fig, ax = plt.subplots(2, 2, figsize=[12, 12])
        for n, (path, weight) in enumerate(zip(paths, weights)):
            vis = nx.MultiDiGraph()
            edge_labels = {}
            print(f"Weight of path {weight:.4f}.")
            for edge in path:
                rel_id = edge[2]
                if rel_id >= 237:
                    rel_id -= 237
                    annot = "INVERSE "
                else:
                    annot = ""
                head_node, tail_node = (
                    self.entity_vocab[edge[0]],
                    self.entity_vocab[edge[1]],
                )
                print(f"   ({head_node}, {annot}{self.relation_vocab[rel_id]}, {tail_node})")
                head_node, tail_node = head_node.replace(" ", "\n"), tail_node.replace(" ", "\n")
                vis.add_node(head_node)
                vis.add_node(tail_node)
                vis.add_edge(head_node, tail_node)
                edge_labels[head_node, tail_node] = (annot + self.relation_vocab[rel_id].split("/")[-1]).replace(
                    " ", "\n"
                )
            query = nx.MultiDiGraph()
            query.add_nodes_from(
                [
                    self.entity_vocab[head_id].replace(" ", "\n"),
                    self.entity_vocab[tail_id].replace(" ", "\n"),
                ]
            )
            query.add_edge(
                self.entity_vocab[head_id].replace(" ", "\n"),
                self.entity_vocab[tail_id].replace(" ", "\n"),
            )

            # vis.add_edge(self.entity_vocab[head_id], self.entity_vocab[tail_id]
            # edge_labels[self.entity_vocab[edge[0]], self.entity_vocab[edge[1]]] = annot + self.relation_vocab[rel_id].split("/")[-1]
            pos = nx.spring_layout(vis)
            nx.draw(vis, pos, with_labels=True, ax=ax[n // 2][n % 2], font_size=10)
            nx.draw_networkx_edge_labels(
                vis,
                pos,
                edge_labels=edge_labels,
                font_color="red",
                ax=ax[n // 2][n % 2],
                font_size=8,
            )
            nx.draw_networkx_edges(
                query,
                pos,
                ax=ax[n // 2][n % 2],
                connectionstyle="arc3,rad=-0.5",
                style="dashed",
            )

        plt.axis("off")
        plt.show()


def match_string(query, choices):
    scores = np.zeros(len(choices))
    s = SequenceMatcher()
    s.set_seq1(query)
    for n, choice in enumerate(choices):
        s.set_seq2(choice)
        scores[n] = s.ratio()
    ind = np.argmax(scores)
    return ind, choices[ind]


@torch.no_grad()
def beam_search_distance(graph, edge_grads, head_id, tail_id, num_nodes, num_beam=10):
    # beam search the top-k distance from h to t (and to every other node)
    input = torch.full((num_nodes, num_beam), float("-inf"))
    input[head_id, 0] = 0
    edge_mask = (graph[:, 0] != tail_id) * (graph[:, 1] != head_id)

    distances = []
    back_edges = []
    for edge_grad in edge_grads:
        # we don't allow any path goes out of t once it arrives at t
        node_in = graph[edge_mask, 0]
        node_out = graph[edge_mask, 1]
        relation = graph[edge_mask, 2]
        edge_grad = edge_grad[edge_mask]

        message = input[node_in] + edge_grad.unsqueeze(-1)  # (num_edges, num_beam)
        # (num_edges, num_beam, 3)
        msg_source = torch.stack([node_in, node_out, relation], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)

        # (num_edges, num_beam)
        is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & (
            msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)
        ).all(dim=-1)
        # pick the first occurrence as the ranking in the previous node's beam
        # this makes deduplication easier later
        # and store it in msg_source
        is_duplicate = is_duplicate.float() - torch.arange(num_beam, dtype=torch.float, device=message.device) / (
            num_beam + 1
        )
        prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
        msg_source = torch.cat([msg_source, prev_rank], dim=-1)  # (num_edges, num_beam, 4)

        node_out, order = node_out.sort()
        node_out_set = torch.unique(node_out)
        # sort messages w.r.t. node_out
        message = message[order].flatten()  # (num_edges * num_beam)
        msg_source = msg_source[order].flatten(0, -2)  # (num_edges * num_beam, 4)
        size = node_out.bincount(minlength=num_nodes)
        msg2out = size_to_index(size[node_out_set] * num_beam)
        # deduplicate messages that are from the same source and the same beam
        is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
        is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
        message = message[~is_duplicate]
        msg_source = msg_source[~is_duplicate]
        msg2out = msg2out[~is_duplicate]
        size = msg2out.bincount(minlength=len(node_out_set))

        if not torch.isinf(message).all():
            # take the topk messages from the neighborhood
            # distance: (len(node_out_set) * num_beam)
            distance, rel_index = scatter_topk(message, size, k=num_beam)
            abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
            # store msg_source for backtracking
            back_edge = msg_source[abs_index]  # (len(node_out_set) * num_beam, 4)
            distance = distance.view(len(node_out_set), num_beam)
            back_edge = back_edge.view(len(node_out_set), num_beam, 4)
            # scatter distance / back_edge back to all nodes
            distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes)  # (num_nodes, num_beam)
            back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes)  # (num_nodes, num_beam, 4)
        else:
            distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
            back_edge = torch.zeros(num_nodes, num_beam, 4, dtype=torch.long, device=message.device)

        distances.append(distance)
        back_edges.append(back_edge)
        input = distance

    return distances, back_edges


def topk_average_length(distances, back_edges, t_index, k=10):
    # backtrack distances and back_edges to generate the paths
    paths = []
    average_lengths = []

    for i in range(len(distances)):
        distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
        back_edge = back_edges[i][t_index].flatten(0, -2)[order]
        for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
            if d == float("-inf"):
                break
            path = [(h, t, r)]
            for j in range(i - 1, -1, -1):
                h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                path.append((h, t, r))
            paths.append(path[::-1])
            average_lengths.append(d / len(path))

    if paths:
        average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

    return paths, average_lengths


def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask]  # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index
