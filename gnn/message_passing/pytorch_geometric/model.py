# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 Weihua Hu
# This file has been modified by Graphcore

import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn.models import MLP


class GIN(nn.Module):
    """
    Graph Isomorphism Network modified to take into account fixed size
    tensor inputs created with padding.

    params:
        in_channels (int): number of features each node is represented by
        hidden_channels (int): number of hidden units for all MLP layers
        out_channels (int): num of hidden units in the output of the network
        num_conv_layers (int): number of GINConv layers in the network
        num_mlp_layers (int): number of hidden layers in MLP
        batch_size (int): maximum number of graphs in a batch
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_conv_layers, num_mlp_layers, batch_size):
        super().__init__()

        self.batch_size = batch_size

        # `num_conv_layers` layers for AGGREGATE and COMBINE
        self.hop_k_gin_layers = nn.ModuleList()

        # linear READOUT nets for (sum) graph pooling
        # the first pooling occurs on the input nodes' features (0-hop)
        self.hop_k_readout_layers = nn.ModuleList([nn.Linear(in_features=in_channels, out_features=out_channels)])

        for k_hop in range(num_conv_layers):
            phi = MLP(
                in_channels=in_channels if k_hop == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_mlp_layers,
                act="relu",
                norm="layer_norm",
                plain_last=False,
            )

            # performs the initial (sum) neighbour pooling + (1-eps) * hv, then applies phi
            # i.e phi o f = MLP((1-eps)*hv + (sum) neighbour k_hop representation)
            self.hop_k_gin_layers.append(GINConv(nn=phi, eps=0, train_eps=False))

            # READOUT is performed on each k_hop node representation for each graph
            self.hop_k_readout_layers.append(nn.Linear(in_features=hidden_channels, out_features=out_channels))

    def forward(self, x, edge_index, batch, graphs_mask=None, target=None):
        # perform k-hop aggregation using GINConv, and return all layer outputs
        hop_k_outputs = [x]
        h = x
        for gin_layer in self.hop_k_gin_layers:
            h = gin_layer(h, edge_index)
            hop_k_outputs.append(h)

        # perform readout over all nodes in each graph in each layer
        score_over_layer = torch.zeros((1))
        for i, linear in enumerate(self.hop_k_readout_layers):
            pooled_h = global_add_pool(x=hop_k_outputs[i], batch=batch, size=self.batch_size)
            # compute scores
            score_over_layer = score_over_layer + nn.functional.dropout(linear(pooled_h), training=self.training)

        if self.training:
            # Mask out the padded graphs
            target = torch.where(graphs_mask, target, -100)
            # Compute loss
            loss = nn.functional.cross_entropy(score_over_layer, target, reduction="sum") / sum(graphs_mask)
            return score_over_layer, loss

        return score_over_layer
