# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Includes derived work from https://github.com/KiddoZhu/NBFNet-PyG
#   Copyright (c) 2021 MilaGraph
#   Licensed under the MIT License

import torch
import os
import copy
from torch.utils.data import DataLoader, IterableDataset, default_collate
from torch.nn import functional as F
import numpy as np
from typing import Optional

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR

import nbfnet_utils


class IndRelLinkPredDataset(InMemoryDataset):
    def __init__(
        self, root, name, version, add_inverse_train=True, add_inverse_test=True, transform=None, pre_transform=None
    ):
        self.name = name
        self.version = version
        self.add_inverse_train = add_inverse_train
        self.add_inverse_test = add_inverse_test
        assert name in ["FB15k-237"]
        assert version in ["v1", "v2", "v3", "v4"]
        self.inv_train_entity_vocab = {}
        self.inv_test_entity_vocab = {}
        self.urls = {
            "FB15k-237": [
                f"https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_{version}_ind/train.txt",
                f"https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_{version}_ind/test.txt",
                f"https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_{version}/train.txt",
                f"https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_{version}/valid.txt",
                f"{RelLinkPredDataset.urls['FB15k-237']}/relations.dict",
            ],
        }
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        annot = ""
        if self.add_inverse_train:
            annot += "_inv_train"
        if self.add_inverse_test:
            annot += "_inv_test"
        return os.path.join(self.root, self.name, self.version, "processed" + annot)

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return ["train_ind.txt", "test_ind.txt", "train.txt", "valid.txt", "relations.dict"]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:4]
        triplets = []
        num_samples = []

        with open(os.path.join(self.raw_dir, "relations.dict")) as file:
            lines = [row.split("\t") for row in file.read().split("\n")[:-1]]
            inv_relation_vocab = {key: int(value) for value, key in lines}

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in self.inv_train_entity_vocab:
                        self.inv_train_entity_vocab[h_token] = len(self.inv_train_entity_vocab)
                    h = self.inv_train_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in self.inv_train_entity_vocab:
                        self.inv_train_entity_vocab[t_token] = len(self.inv_train_entity_vocab)
                    t = self.inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in self.inv_test_entity_vocab:
                        self.inv_test_entity_vocab[h_token] = len(self.inv_test_entity_vocab)
                    h = self.inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in self.inv_test_entity_vocab:
                        self.inv_test_entity_vocab[t_token] = len(self.inv_test_entity_vocab)
                    t = self.inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = len(inv_relation_vocab)

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_target_index = edge_index[:, train_slice]
        train_target_type = edge_type[train_slice]
        valid_target_index = edge_index[:, valid_slice]
        valid_target_type = edge_type[valid_slice]
        test_target_index = edge_index[:, test_slice]
        test_target_type = edge_type[test_slice]
        # add flipped training triples
        if self.add_inverse_train:
            train_target_index = torch.cat([train_target_index, train_target_index.flip(0)], dim=-1)
            train_target_type = torch.cat([train_target_type, train_target_type + num_relations])

        # add flipped validation and test triples
        if self.add_inverse_test:
            valid_target_index = torch.cat([valid_target_index, valid_target_index.flip(0)], dim=-1)
            valid_target_type = torch.cat([valid_target_type, valid_target_type + num_relations])
            test_target_index = torch.cat([test_target_index, test_target_index.flip(0)], dim=-1)
            test_target_type = torch.cat([test_target_type, test_target_type + num_relations])

        train_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=len(self.inv_train_entity_vocab),
            target_edge_index=train_target_index,
            target_edge_type=train_target_type,
        )
        valid_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=len(self.inv_train_entity_vocab),
            target_edge_index=valid_target_index,
            target_edge_type=valid_target_type,
        )
        test_data = Data(
            edge_index=test_fact_index,
            edge_type=test_fact_type,
            num_nodes=len(self.inv_test_entity_vocab),
            target_edge_index=test_target_index,
            target_edge_type=test_target_type,
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name


def build_dataset(
    name: str, path: str, version: str = None, add_inverse_train: bool = True, add_inverse_test: bool = True
):
    if name == "FB15k-237":
        dataset = RelLinkPredDataset(name=name, root=path)
        data = dataset.data
        train_target_index = data.train_edge_index
        train_target_type = data.train_edge_type
        valid_target_index = data.valid_edge_index
        valid_target_type = data.valid_edge_type
        test_target_index = data.test_edge_index
        test_target_type = data.test_edge_type

        # add flipped training triples
        if add_inverse_train:
            train_target_index = torch.cat([train_target_index, train_target_index.flip(0)], dim=-1)
            train_target_type = torch.cat([train_target_type, train_target_type + dataset.num_relations // 2])

        # add flipped validation and test triples
        if add_inverse_test:
            valid_target_index = torch.cat([valid_target_index, valid_target_index.flip(0)], dim=-1)
            valid_target_type = torch.cat([valid_target_type, valid_target_type + dataset.num_relations // 2])
            test_target_index = torch.cat([test_target_index, test_target_index.flip(0)], dim=-1)
            test_target_type = torch.cat([test_target_type, test_target_type + dataset.num_relations // 2])

        train_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            num_nodes=data.num_nodes,
            target_edge_index=train_target_index,
            target_edge_type=train_target_type,
        )
        valid_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            num_nodes=data.num_nodes,
            target_edge_index=valid_target_index,
            target_edge_type=valid_target_type,
        )
        test_data = Data(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            num_nodes=data.num_nodes,
            target_edge_index=test_target_index,
            target_edge_type=test_target_type,
        )
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    elif name == "WN18RR":
        dataset = WordNet18RR(root=path)
        # convert wn18rr into the same format as fb15k-237
        data = dataset.data
        num_nodes = int(data.edge_index.max()) + 1
        num_relations = int(data.edge_type.max()) + 1
        edge_index = data.edge_index[:, data.train_mask]
        edge_type = data.edge_type[data.train_mask]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        edge_type = torch.cat([edge_type, edge_type + num_relations])
        if add_inverse_train:
            raise NotImplementedError
        if add_inverse_test:
            raise NotImplementedError
        train_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            target_edge_index=data.edge_index[:, data.train_mask],
            target_edge_type=data.edge_type[data.train_mask],
        )
        valid_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            target_edge_index=data.edge_index[:, data.val_mask],
            target_edge_type=data.edge_type[data.val_mask],
        )
        test_data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            target_edge_index=data.edge_index[:, data.test_mask],
            target_edge_type=data.edge_type[data.test_mask],
        )
        dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
        dataset.num_relations = num_relations * 2
    elif name.startswith("Ind"):
        dataset = IndRelLinkPredDataset(
            name=name[3:],
            root=path,
            version=version,
            add_inverse_train=add_inverse_train,
            add_inverse_test=add_inverse_test,
        )
    else:
        raise ValueError("Unknown dataset `%s`" % name)
    return dataset


class DataWrapper(IterableDataset):
    def __init__(self, data):
        super(DataWrapper).__init__()
        self.data = data

    def __len__(self):
        return self.data.n_batches

    def __iter__(self):
        return self.data.batches()


class NBFData:
    def __init__(
        self,
        data: Data,
        batch_size: int,
        is_training: bool,
        num_relations: Optional[int] = None,
        num_negatives: Optional[int] = None,
        check_negatives: Optional[bool] = True,
        edge_dropout: Optional[float] = 0.0,
    ):

        self.batch_size = batch_size
        self.is_training = is_training
        self.num_relations = num_relations

        self.num_negatives = num_negatives
        self.check_negatives = check_negatives
        self.edge_dropout = edge_dropout

        self.data = transpose_dataset(data)
        self.num_nodes = data.num_nodes
        self.num_edges = len(data.edge_type)
        self.graph = torch.cat([self.data.edge_index, self.data.edge_type.unsqueeze(1)], dim=1)
        self.dataloader = DataLoader(
            torch.cat([self.data.target_edge_index, self.data.target_edge_type.unsqueeze(1)], dim=1),
            batch_size,
            drop_last=False,
            shuffle=is_training,
        )

    @staticmethod
    def pad(value, shape, pad_value):
        dims = list(zip(value.shape, shape))
        assert all(
            actual <= target for actual, target in dims
        ), f"original shape {value.shape} larger than target {shape}"
        padding = []
        for actual, target in reversed(dims):
            padding.extend([0, target - actual])
        return F.pad(value, padding, value=pad_value)

    def batches(self):
        for n, batch in enumerate(self.dataloader):
            if self.num_negatives:
                head_id, tail_id, relation_id = nbfnet_utils.negative_sampling(
                    batch=batch,
                    graph=self.graph,
                    num_nodes=self.num_nodes,
                    num_negative=self.num_negatives,
                    strict=self.check_negatives,
                )
                num_negative = self.num_negatives
            else:
                head_id, tail_id, relation_id = nbfnet_utils.all_negative(batch, self.num_nodes)
                num_negative = self.num_nodes - 1

            if self.is_training:
                graph = self.remove_easy_edges(self.graph, head_id, tail_id, relation_id, self.edge_dropout)
            else:
                graph = self.graph

            yield dict(
                graph=graph + 1,
                num_nodes=self.num_nodes + 1,
                head_id=self.pad(head_id + 1, [self.batch_size], 0),
                tail_id=self.pad(tail_id + 1, [self.batch_size, 1 + num_negative], 0),
                relation_id=self.pad(relation_id + 1, [self.batch_size], 0),
            )

    def remove_easy_edges(
        self,
        graph: torch.Tensor,
        head_id: torch.Tensor,
        tail_id: torch.Tensor,
        relation_id: torch.Tensor,
        edge_dropout: Optional[float] = 0.0,
    ):
        """For a given batch remove direct edges between head and tail entities and
        their inverse from the graph
        :param graph: The full set of triples (head, tail, relation). Shape [num_triples, 3]
        :param head_id: Head of edges to be removed. Shape [batch_size]
        :param tail_id: Tail of edges to be removed. Shape [batch_size, 1 + num_negatives]
        :param relation_id: Relation of edges to be removed.  Shape [batch_size]
        :param edge_dropout: Optional dropout rate for remaining edges.
        :return: graph with direct edges replaced by padding tokens (0, 0, 0)
        """
        if not self.check_negatives:
            num_tails = tail_id.shape[1]
            head_id = head_id.repeat(num_tails)
            tail_id = tail_id.t().flatten()
            relation_id = relation_id.repeat(num_tails)
        else:
            tail_id = tail_id[:, 0]

        # Also remove inverse edges
        head_id_ext = torch.cat([head_id, tail_id], dim=-1)
        tail_id_ext = torch.cat([tail_id, head_id], dim=-1)
        relation_id_ext = torch.cat([relation_id, relation_id + self.num_relations // 2 % self.num_relations], dim=-1)

        to_remove = torch.stack([head_id_ext, tail_id_ext, relation_id_ext], -1)
        id_remove = nbfnet_utils.edge_match(graph, to_remove)[0]
        if edge_dropout:
            id_remove = torch.cat([id_remove, torch.randint(0, self.num_edges, [int(self.num_edges * edge_dropout)])])
        mask_remove = torch.zeros(graph.shape[0], dtype=torch.bool)
        mask_remove[id_remove] = True
        modified_graph = copy.deepcopy(graph)
        modified_graph[mask_remove, :] = 0  # Padding token 0 for node and relation ids
        return modified_graph

    @property
    def n_items(self):
        return len(self.data.target_edge_type)

    @property
    def n_batches(self):
        return int(np.ceil(self.n_items / self.batch_size))


def transpose_dataset(data):
    return Data(
        edge_index=data.edge_index.t(),
        edge_type=data.edge_type,
        num_nodes=data.num_nodes,
        target_edge_index=data.target_edge_index.t(),
        target_edge_type=data.target_edge_type,
    )


def custom_collate(batches: list):
    out = dict()
    for key in batches[0].keys():
        if key == "num_nodes":
            out[key] = [batch[key] for batch in batches]
        else:
            out[key] = default_collate([batch[key] for batch in batches])
    return out
