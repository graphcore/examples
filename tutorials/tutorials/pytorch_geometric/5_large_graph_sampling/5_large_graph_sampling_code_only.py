# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os

dataset_directory = os.getenv("DATASETS_DIR", "data")

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

transform = T.Compose([T.NormalizeFeatures(), T.AddSelfLoops()])

dataset = Planetoid(root=dataset_directory, name="PubMed", transform=transform)
data = dataset[0]  # Access the graph as Data object

print(f"Dataset: {dataset} ")
print(f"Number of graphs: {len(dataset)}: ")
print(f"Number of features: {dataset.num_features} ")
print(f"Number of classes: {dataset.num_classes} ")

print(data)

print(f"Total number of nodes: {data.num_nodes}")
print(f"Total number of edges: {data.num_edges}")

from torch_geometric.loader import ClusterData

num_clusters = 100

cluster_data = ClusterData(data, num_parts=num_clusters, recursive=False, save_dir=dataset_directory)

print(f"The dataset has been split in {len(cluster_data)} clusters")

from torch_geometric.loader import ClusterLoader

clusters_per_batch = 10

dynamic_size_dataloader = ClusterLoader(
    cluster_data,
    batch_size=clusters_per_batch,
)

dynamic_dataloader_iter = iter(dynamic_size_dataloader)
print(f"{next(dynamic_dataloader_iter) = }")
print(f"{next(dynamic_dataloader_iter) = }")

import poptorch
from poptorch_geometric import FixedSizeOptions, OverSizeStrategy

fixed_size_options = FixedSizeOptions.from_loader(dynamic_size_dataloader, sample_limit=10)

print(fixed_size_options)

from poptorch_geometric.cluster_loader import FixedSizeClusterLoader

train_dataloader = FixedSizeClusterLoader(
    cluster_data,
    batch_size=clusters_per_batch,
    fixed_size_options=fixed_size_options,
    over_size_strategy=OverSizeStrategy.TrimNodesAndEdges,
)

train_dataloader_iter = iter(train_dataloader)
print(f"{next(train_dataloader_iter) = }")
print(f"{next(train_dataloader_iter) = }")

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(GCN, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_channels, 64, add_self_loops=False)
        self.conv2 = GCNConv(64, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, train_mask=None, y=None):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=-1)

        if self.training:
            y = torch.where(train_mask, y, -100)
            loss = F.nll_loss(x, y)
            return x, loss
        return x


model = GCN(dataset.num_features, dataset.num_classes)
model

from torchinfo import summary

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)

summary(poptorch_model)

from tqdm import tqdm

num_epochs = 5
train_losses = torch.empty(num_epochs, len(train_dataloader))

for epoch in range(num_epochs):
    bar = tqdm(train_dataloader)
    for i, batch in enumerate(bar):
        _, mini_batch_loss = poptorch_model(batch.x, batch.edge_index, batch.train_mask, batch.y)
        train_losses[epoch, i] = float(mini_batch_loss.mean())
        bar.set_description(f"Epoch {epoch} training loss: {train_losses[epoch, i].item():0.6f}")
        optimizer.zero_grad()  # clear gradients

poptorch_model.detachFromDevice()

import matplotlib.pyplot as plt

plt.plot(train_losses.mean(dim=1))
plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.legend(["Mean training loss per epoch"])
plt.xticks(torch.arange(0, num_epochs, 2))
plt.gcf().set_dpi(150)

from torch_geometric.loader import NeighborLoader

num_neighbors = 10
num_iterations = 2
batch_size = 5

train_loader_sampling = NeighborLoader(
    data,
    shuffle=True,
    num_neighbors=[num_neighbors] * num_iterations,
    batch_size=batch_size,
)

train_sampling_iter = iter(train_loader_sampling)
print(f"{next(train_sampling_iter) = }")
print(f"{next(train_sampling_iter) = }")

sampled_data = next(train_sampling_iter)
print(f"Original graph node index of each target node in the mini-batch: {sampled_data.input_id}")
print(
    f"Original graph node index of each node in the mini-batch: {sampled_data.n_id}"
)  # shows the target nodes are the first 5

fixed_size_options = FixedSizeOptions.from_loader(train_loader_sampling, sample_limit=10)

print(fixed_size_options)

from poptorch_geometric.neighbor_loader import FixedSizeNeighborLoader

train_loader_ipu = FixedSizeNeighborLoader(
    data,
    shuffle=True,
    num_neighbors=[num_neighbors] * num_iterations,
    fixed_size_options=fixed_size_options,
    over_size_strategy=OverSizeStrategy.TrimNodesAndEdges,
    batch_size=batch_size,
)

train_loader_ipu_iter = iter(train_loader_ipu)
print(f"{next(train_loader_ipu_iter) = }")
print(f"{next(train_loader_ipu_iter) = }")
print(f"{next(train_loader_ipu_iter).input_id = }")

from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(GraphSAGE, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = SAGEConv(in_channels, 64, add_self_loops=False)
        self.conv2 = SAGEConv(64, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, train_mask=None, y=None):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=-1)

        if self.training:
            loss = F.nll_loss(
                x[:batch_size], y[:batch_size]
            )  # Select only the target nodes for loss calculation, leave out the padding sub-graph in each sub-graph
            return x, loss
        return x


model_sampling = GraphSAGE(dataset.num_features, dataset.num_classes)
model_sampling

from torchinfo import summary

model_sampling.train()
optimizer = poptorch.optim.Adam(model_sampling.parameters(), lr=0.001)
poptorch_model_sampling = poptorch.trainingModel(model_sampling, optimizer=optimizer)

summary(poptorch_model_sampling)

from tqdm import tqdm

num_epochs = 5
epoch_losses = torch.empty(num_epochs, len(train_loader_ipu))

for epoch in range(num_epochs):
    bar = tqdm(train_loader_ipu)
    for i, batch in enumerate(bar):
        _, mini_batch_loss = poptorch_model_sampling(batch.x, batch.edge_index, batch.train_mask, batch.y)
        epoch_losses[epoch, i] = float(mini_batch_loss.mean())
        bar.set_description(f"Epoch {epoch} training loss: {epoch_losses[epoch, i].item():0.6f}")

import matplotlib.pyplot as plt

plt.plot(epoch_losses.mean(dim=1))
plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.legend(["Mean training loss per epoch"])
plt.grid(True)
plt.xticks(torch.arange(0, num_epochs, 2))
plt.gcf().set_dpi(150)

# Generated:2023-07-03T17:26 Source:5_large_graph_sampling.py SST:0.0.10
