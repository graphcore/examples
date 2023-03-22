# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os

executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-a-worked-example"
dataset_directory = os.getenv("DATASET_DIR", "data")

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

transform = T.Compose([T.NormalizeFeatures(), T.AddSelfLoops()])

dataset = Planetoid(root=dataset_directory, name="Cora", transform=transform)
data = dataset[0]  # Access the citation graph as Data object

print(f"Dataset: {dataset}: ")
print(f"Number of graphs: {len(dataset)}: ")
print(f"Number of features: {dataset.num_features}: ")
print(f"Number of classes: {dataset.num_classes}: ")

print(data)

print(f"{data.num_nodes = }")
print(f"{data.num_edges = }")

from poptorch_geometric.dataloader import DataLoader

dataloader = DataLoader(dataset, batch_size=1)

from torch_geometric import nn

# List all the layers which are a subclass of the MessagePassing layer
attrs = []
for attr in dir(nn):
    try:
        if issubclass(getattr(nn, attr), nn.MessagePassing):
            attrs.append(attr)
    except:
        pass
print(attrs)

from torch_geometric.nn import GCNConv

conv = GCNConv(16, 16, add_self_loops=False)

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16, add_self_loops=False)
        self.conv2 = GCNConv(16, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, y=None, train_mask=None):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.log_softmax(x, dim=1)

        if self.training:
            y = torch.where(train_mask, y, -100)
            loss = F.nll_loss(x, y)
            return x, loss
        return x


print(f"{dataset.num_node_features = }")
print(f"{dataset.num_classes = }")

in_channels = dataset.num_node_features
out_channels = dataset.num_classes

model = GCN(in_channels, out_channels)
model.train()

import poptorch

optimizer = poptorch.optim.Adam(model.parameters(), lr=0.001)
poptorch_options = poptorch.Options().enableExecutableCaching(executable_cache_dir)
poptorch_model = poptorch.trainingModel(model, poptorch_options, optimizer=optimizer)

from tqdm import tqdm

losses = []

for epoch in tqdm(range(100)):
    bar = tqdm(dataloader)
    for data in bar:
        _, loss = poptorch_model(data.x, data.edge_index, y=data.y, train_mask=data.train_mask)
        bar.set_description(f"Epoch {epoch} loss: {loss:0.6f}")
        losses.append(loss)

poptorch_model.detachFromDevice()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(list(range(len(losses))), losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.grid(True)

model.eval()
poptorch_inf_model = poptorch.inferenceModel(model, options=poptorch_options)

data = next(iter(dataset))
logits = poptorch_inf_model(data.x, data.edge_index)
poptorch_inf_model.detachFromDevice()

pred = logits.argmax(dim=1)
pred

correct_results = pred[data.val_mask] == data.y[data.val_mask]
accuracy = int(correct_results.sum()) / int(data.val_mask.sum())
print(f"Validation accuracy: {accuracy:.2%}")

correct_results = pred[data.test_mask] == data.y[data.test_mask]
accuracy = int(correct_results.sum()) / int(data.test_mask.sum())
print(f"Test accuracy: {accuracy:.2%}")
