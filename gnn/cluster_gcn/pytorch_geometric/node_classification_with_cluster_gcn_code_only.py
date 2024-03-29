# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

# THIS FILE IS AUTOGENERATED. Rerun SST after editing source file: node_classification_with_cluster_gcn.py

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import poptorch
import torch
import torch.nn.functional as F
from poptorch_geometric import FixedSizeOptions, OverSizeStrategy
from poptorch_geometric.cluster_loader import FixedSizeClusterLoader
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.datasets import Reddit
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

poptorch.setLogLevel("ERR")
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-clustergcn"
dataset_directory = os.getenv("DATASETS_DIR", "data")

reddit_root = osp.join(dataset_directory, "Reddit")
dataset = Reddit(reddit_root)

len(dataset)

dataset[0]

total_num_clusters = 1500

cluster_data = ClusterData(dataset[0], num_parts=total_num_clusters, recursive=False, save_dir=reddit_root)

len(cluster_data)

cluster_data[0]

num_nodes_per_cluster = []
num_edges_per_cluster = []

for cluster in cluster_data:
    num_nodes_per_cluster.append(cluster.y.shape[0])
    num_edges_per_cluster.append(cluster.edge_index.shape[1])

plt.hist(np.array(num_nodes_per_cluster), 20)
plt.xlabel("Number of nodes per cluster")
plt.ylabel("Counts")
plt.title("Histogram of nodes in each cluster")
plt.show()

plt.hist(np.array(num_edges_per_cluster), 20)
plt.xlabel("Number of edges per cluster")
plt.ylabel("Counts")
plt.title("Histogram of edges in each cluster")
plt.show()

clusters_per_batch = 6

dynamic_size_dataloader = ClusterLoader(
    cluster_data,
    batch_size=clusters_per_batch,
)

fixed_size_options = FixedSizeOptions.from_loader(dynamic_size_dataloader, sample_limit=10)
print(fixed_size_options)

train_dataloader = FixedSizeClusterLoader(
    cluster_data,
    batch_size=clusters_per_batch,
    fixed_size_options=fixed_size_options,
    over_size_strategy=OverSizeStrategy.TrimNodesAndEdges,
    num_workers=8,
)

train_dataloader_iter = iter(train_dataloader)

print(next(train_dataloader_iter))
print(next(train_dataloader_iter))


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = SAGEConv(in_channels, 128)
        self.conv_2 = SAGEConv(128, out_channels)

    def forward(self, x, edge_index, mask=None, target=None):
        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv_2(x, edge_index)
        out = F.log_softmax(x, dim=-1)

        if self.training:
            # Mask out the nodes we don't care about
            target = torch.where(mask, target, -100)
            return out, F.nll_loss(out, target)
        return out


options = poptorch.Options()
options.deviceIterations(4)
options.outputMode(poptorch.OutputMode.All)
options.enableExecutableCaching(executable_cache_dir)

train_dataloader = FixedSizeClusterLoader(
    cluster_data,
    batch_size=clusters_per_batch,
    fixed_size_options=fixed_size_options,
    over_size_strategy=OverSizeStrategy.TrimNodesAndEdges,
    num_workers=8,
    options=options,
)

train_dataloader_iter = iter(train_dataloader)

print(next(train_dataloader_iter))
print(next(train_dataloader_iter))

model = Net(dataset.num_features, dataset.num_classes)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
poptorch_model = poptorch.trainingModel(model, optimizer=optimizer, options=options)

num_epochs = 10
train_losses = torch.empty(num_epochs, len(train_dataloader))

for epoch in range(num_epochs):
    bar = tqdm(train_dataloader)
    for i, data in enumerate(bar):
        # Performs forward pass, loss function evaluation,
        # backward pass and weight update in one go on the device.
        _, mini_batch_loss = poptorch_model(data.x, data.edge_index, data.train_mask, data.y)
        train_losses[epoch, i] = float(mini_batch_loss.mean())
        bar.set_description(f"Epoch {epoch} training loss: {train_losses[epoch, i].item():0.6f}")

poptorch_model.detachFromDevice()

plt.plot(train_losses.mean(dim=1))
plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.legend(["Training loss"])
plt.grid(True)
plt.xticks(torch.arange(0, num_epochs, 2))
plt.gcf().set_dpi(150)

"""
data = dataset[0]

model = Net(dataset.num_features, dataset.num_classes)
model.load_state_dict(poptorch_model.state_dict())
model.eval()
out = model.forward(data.x, data.edge_index)
y_pred = out.argmax(dim=-1)

accs = []
for mask in [data.val_mask, data.test_mask]:
    correct = y_pred[mask].eq(data.y[mask]).sum().item()
    accs.append(correct / mask.sum().item())

print("Validation accuracy: {accs[0]}")
print("Test accuracy: {accs[1]}")
"""

# Generated:2023-05-23T13:18 Source:node_classification_with_cluster_gcn.py SST:0.0.10
