# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import Linear, SAGEConv

from loss import weighted_cross_entropy


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        assert num_layers > 1
        self.convs = ModuleList([SAGEConv((-1, -1), hidden_channels) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x


class Model(torch.nn.Module):
    def __init__(
        self,
        hetero_gnn,
        embedding_size,
        out_channels,
        node_types,
        num_nodes_per_type,
        class_weight=None,
        full_batch=False,
    ):
        super().__init__()
        self.hetero_gnn = hetero_gnn
        self.embedding = nn.ModuleDict(
            {
                node_type: nn.Embedding(num_nodes_per_type[node_type], embedding_size)
                for node_type in node_types
                if node_type != "transaction"
            }
        )
        self.linear = Linear(-1, out_channels)
        self.node_types = node_types
        self.full_batch = full_batch
        self.class_weight = class_weight

    def forward(self, x_dict, edge_index_dict, batch_size=None, n_id_dict=None, target=None, mask=None):
        for node_type in self.node_types:
            if node_type != "transaction":
                if self.full_batch:
                    x_dict[node_type] = self.embedding[node_type].weight
                else:
                    assert n_id_dict is not None, "If using a sampled batch, `n_id_dict` must be provided."
                    x_dict[node_type] = self.embedding[node_type](n_id_dict[node_type])

        x_dict = self.hetero_gnn(x_dict, edge_index_dict)
        out = self.linear(x_dict["transaction"])
        if self.training:
            if not self.full_batch:
                assert batch_size is not None, "If using a sampled batch, `batch_size` must be provided."
                mask = (target * 0).bool()
                mask[:batch_size] = 1
            loss = weighted_cross_entropy(out, target, mask, self.class_weight)
            return out, loss
        return out
