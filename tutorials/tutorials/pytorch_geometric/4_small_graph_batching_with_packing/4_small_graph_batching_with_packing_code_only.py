# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os

executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-packing"
dataset_directory = os.getenv("DATASET_DIR", "data")

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root=f"{dataset_directory}/TUDataset", name="MUTAG")

print(f"{len(dataset) = }")
print(f"{dataset.num_features = }")
print(f"{dataset.num_edge_features = }")
print(f"{dataset.num_classes = }")

first_molecule = dataset[0]

print(f"{first_molecule.num_nodes = }")
print(f"{first_molecule.num_edges = }")
print(f"{first_molecule.y = }")

second_molecule = dataset[1]

print(f"{second_molecule.num_nodes = }")
print(f"{second_molecule.num_edges = }")
print(f"{second_molecule.y = }")

import torch

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

from torch_geometric.data.summary import Summary

dataset_summary = Summary.from_dataset(dataset)
dataset_summary

max_num_graphs_per_batch = 10
max_num_nodes_per_batch = 400
max_num_edges_per_batch = 800

print("Maximum number of graphs per mini-batch:", max_num_graphs_per_batch)
print("Maximum number of nodes per mini-batch:", max_num_nodes_per_batch)
print("Maximum number of edges per mini-batch:", max_num_edges_per_batch)

from poptorch_geometric.dataloader import FixedSizeDataLoader

train_dataloader = FixedSizeDataLoader(
    train_dataset,
    batch_size=max_num_graphs_per_batch,
    num_nodes=max_num_nodes_per_batch,
    num_edges=max_num_edges_per_batch,
    collater_args=dict(add_masks_to_batch=True),
)

train_dataloader_iter = iter(train_dataloader)

first_sample = next(train_dataloader_iter)
second_sample = next(train_dataloader_iter)

print(f"{first_sample = }")
print(f"{second_sample = }")

print(first_sample.graphs_mask)
print(second_sample.graphs_mask)

first_sample.nodes_mask

number_of_real_nodes_in_batch = int(first_sample.nodes_mask.sum())
print(f"{number_of_real_nodes_in_batch = }")
print(f"{max_num_nodes_per_batch = }")

number_of_real_nodes_in_batch / max_num_nodes_per_batch


def get_node_packing_efficiency(train_dataloader, total_number_of_nodes_in_batch):
    packing_efficiency_per_pack = []

    for data in train_dataloader:
        number_of_real_nodes_in_batch = int(sum(data.nodes_mask))
        total_number_of_nodes_in_batch = len(data.nodes_mask)
        packing_efficiency_per_pack.append(number_of_real_nodes_in_batch / total_number_of_nodes_in_batch)

    return sum(packing_efficiency_per_pack) / len(packing_efficiency_per_pack)


packing_efficiency = get_node_packing_efficiency(train_dataloader, max_num_nodes_per_batch)
print(f"{packing_efficiency = :.2%}")

max_num_graphs_per_batch = 10

max_num_nodes_per_batch = int(dataset_summary.num_nodes.mean * max_num_graphs_per_batch)
max_num_edges_per_batch = int(dataset_summary.num_edges.mean * max_num_graphs_per_batch)

print("Maximum number of graphs per batch:", max_num_graphs_per_batch)
print("Maximum number of nodes per batch:", max_num_nodes_per_batch)
print("Maximum number of edges per batch:", max_num_edges_per_batch)

train_dataloader = FixedSizeDataLoader(
    train_dataset,
    batch_size=max_num_graphs_per_batch,
    num_nodes=max_num_nodes_per_batch,
    num_edges=max_num_edges_per_batch,
    collater_args=dict(add_masks_to_batch=True),
)

packing_efficiency = get_node_packing_efficiency(train_dataloader, max_num_nodes_per_batch)
print(f"{packing_efficiency = :.2%}")

for i, batch in enumerate(train_dataloader):
    num_of_real_graphs_in_batch = int(batch.graphs_mask.sum())
    print(f"Mini-batch {i}, number of real graphs {num_of_real_graphs_in_batch}")

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class GcnForPacking(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, batch_size):
        super(GcnForPacking, self).__init__()
        self.conv = GCNConv(dataset.num_node_features, hidden_channels, add_self_loops=False)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.batch_size = batch_size

    def forward(self, x, edge_index, batch, graphs_mask, y):
        # 1. Obtain node embeddings
        x = self.conv(x, edge_index).relu()
        # 2. Pooling layer
        x = global_mean_pool(x, batch, size=self.batch_size)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        y = torch.where(graphs_mask, y, -100)

        if self.training:
            return F.cross_entropy(x, y, reduction="sum") / sum(graphs_mask)
        return x


import poptorch

# Initialise model and convert the model to a poptorch model
opts = poptorch.Options().enableExecutableCaching(executable_cache_dir)
model = GcnForPacking(hidden_channels=64, dataset=dataset, batch_size=max_num_graphs_per_batch)
optim = poptorch.optim.Adam(model.parameters(), lr=0.01)
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optim)
poptorch_model

poptorch_model.train()
loss_per_epoch = []

for epoch in range(0, 10):
    total_loss = 0

    for data in train_dataloader:
        loss = poptorch_model(data.x, data.edge_index, data.batch, data.graphs_mask, data.y)  # Forward pass.
        total_loss += loss
        optim.zero_grad()  # Clear gradients.

    loss_this_epoch = total_loss / len(dataset)
    loss_per_epoch.append(loss_this_epoch)
    print("Epoch:", epoch, " Training Loss: ", loss_this_epoch)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(loss_per_epoch)
plt.title("Loss per epoch using the Packed Dataloader")
plt.xlabel("Epoch")
plt.ylabel("Loss")

test_dataloader = FixedSizeDataLoader(
    test_dataset,
    batch_size=max_num_graphs_per_batch,
    num_nodes=max_num_nodes_per_batch,
    num_edges=max_num_edges_per_batch,
    collater_args=dict(add_masks_to_batch=True),
)

inf_model = poptorch.inferenceModel(model, options=poptorch.Options().enableExecutableCaching(executable_cache_dir))
inf_model.eval()

correct = 0

for data in test_dataloader:
    out = inf_model(data.x, data.edge_index, data.batch, data.graphs_mask, data.y)  # Forward pass.
    pred = out.argmax(dim=1)
    correct += int(((pred == data.y) * data.graphs_mask).sum())

accuracy = correct / len(train_dataset)
print(f"{accuracy = }")
