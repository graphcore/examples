# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os

executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-heterogeneous"
dataset_directory = os.getenv("DATASETS_DIR", "data")

from torch_geometric.datasets import IMDB

dataset = IMDB(root=f"{dataset_directory}/IMDB")

data = dataset[0]
data

import torch

classes = torch.unique(data["movie"].y)
num_classes = len(classes)
classes, num_classes

from torch_geometric.transforms import RemoveIsolatedNodes

data = data.subgraph({"movie": torch.arange(0, 1000)})
data = RemoveIsolatedNodes()(data)
data

import torch
from torch_geometric.nn import SAGEConv


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 64)
        self.conv2 = SAGEConv((-1, -1), 64)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


from torch_geometric.nn import to_hetero

# Initialize the model
model = Model()
# Convert the model to a heterogeneous model
model = to_hetero(model, data.metadata(), aggr="sum")
model

# Initialize lazy modules.
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)

import torch.nn.functional as F


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_dict, edge_index_dict, target=None, train_mask=None):
        out = self.model(x_dict, edge_index_dict)
        if self.training:
            target = torch.where(train_mask, target, -100)
            loss = F.cross_entropy(out["movie"], target)
            return out, loss
        return out


# Include loss in model
model = ModelWithLoss(model)

import poptorch

# Set up training
model.train()

# Initialise model and convert the model to a PopTorch model
opts = poptorch.Options().enableExecutableCaching(executable_cache_dir)
optim = poptorch.optim.Adam(model.parameters(), lr=0.01)
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optim)

# Train
for _ in range(3):
    out, loss = poptorch_model(
        data.x_dict,
        data.edge_index_dict,
        target=data["movie"].y,
        train_mask=data["movie"].train_mask,
    )
    print(f"{loss = }")

from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("movie", "to", "director"): SAGEConv((-1, -1), hidden_channels),
                    ("director", "to", "movie"): SAGEConv((-1, -1), hidden_channels),
                    ("movie", "to", "actor"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ("actor", "to", "movie"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, target=None, train_mask=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        out = self.lin(x_dict["movie"])

        if self.training:
            target = torch.where(train_mask, target, -100)
            loss = F.cross_entropy(out, target)
            return out, loss
        return out


model = HeteroGNN(hidden_channels=64, out_channels=num_classes, num_layers=2)

# Initialize lazy modules.
with torch.no_grad():
    out = model(
        data.x_dict,
        data.edge_index_dict,
        target=data["movie"].y,
        train_mask=data["movie"].train_mask,
    )

# Set up training
model.train()

# Initialise model and convert the model to a PopTorch model
opts = poptorch.Options().enableExecutableCaching(executable_cache_dir)
optim = poptorch.optim.Adam(model.parameters(), lr=0.01)
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optim)

# Train
for _ in range(3):
    out, loss = poptorch_model(
        data.x_dict,
        data.edge_index_dict,
        target=data["movie"].y,
        train_mask=data["movie"].train_mask,
    )
    print(f"{loss = }")

from torch_geometric.loader import NeighborLoader

train_loader = NeighborLoader(
    data,
    num_neighbors=[5] * 2,
    batch_size=5,
    input_nodes=("movie", data["movie"].train_mask),
)

next(iter(train_loader))

from poptorch_geometric import FixedSizeOptions

fixed_size_options = FixedSizeOptions(
    num_nodes=1000,
    num_edges=1000,
)
fixed_size_options

from poptorch_geometric import OverSizeStrategy
from poptorch_geometric.neighbor_loader import FixedSizeNeighborLoader


fixed_size_train_loader = FixedSizeNeighborLoader(
    data,
    num_neighbors=[5] * 2,
    batch_size=5,
    input_nodes=("movie", data["movie"].train_mask),
    fixed_size_options=fixed_size_options,
    over_size_strategy=OverSizeStrategy.TrimNodesAndEdges,
)

next(iter(fixed_size_train_loader))

fixed_size_options = FixedSizeOptions(
    num_nodes={"movie": 500, "director": 100, "actor": 300},
    num_edges={
        ("movie", "to", "director"): 100,
        ("movie", "to", "actor"): 200,
        ("director", "to", "movie"): 100,
        ("actor", "to", "movie"): 200,
    },
)
fixed_size_options

fixed_size_options = FixedSizeOptions.from_loader(train_loader)
fixed_size_options

fixed_size_train_loader = FixedSizeNeighborLoader(
    data,
    num_neighbors=[5] * 2,
    batch_size=5,
    input_nodes=("movie", data["movie"].train_mask),
    fixed_size_options=fixed_size_options,
    over_size_strategy=OverSizeStrategy.TrimNodesAndEdges,
)

next(iter(fixed_size_train_loader))

# Generated:2023-06-27T10:07 Source:6_heterogeneous_graphs.py SST:0.0.10
