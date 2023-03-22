# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os

executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-padding"
dataset_directory = os.getenv("DATASET_DIR", "data")

from torch_geometric.datasets import QM9

dataset = QM9(root=dataset_directory)
print(dataset)
print(dataset[2])

import ipywidgets
import py3Dmol
from ipywidgets import interact
from periodictable import elements


def MolTo3DView(mol, loader_on, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """

    assert style in ("line", "stick", "sphere", "carton")
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mol, "mol")
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {"opacity": opacity})
    viewer.zoomTo()
    return viewer


def molecule_converter(datum):
    num_atoms = int(datum.z.numel())
    xyz = f"{num_atoms}\n\n"
    for i in range(num_atoms):
        sym = elements[datum.z[i].item()].symbol
        r = datum.pos[i, :].tolist()
        line = [sym] + [f"{i: 0.08f}" for i in r]
        line = "\t".join(line)
        xyz += f"{line}\n"
    return xyz


def conf_viewer(idx):
    mol = smi[idx]
    return MolTo3DView(mol, loader_on=True, size=(300, 300)).show()


bs = 12
data_chunk = dataset[0:bs]
smi = [molecule_converter(i) for i in data_chunk]
interact(conf_viewer, idx=ipywidgets.IntSlider(min=0, max=bs - 1, step=1))

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=4)
it = iter(loader)
next(it), next(it), next(it)

for i, batch in zip(range(3), loader):
    print(f"Mini-batch {i} has an x tensor of shape: {batch.x.shape}")

from torch_geometric.data.summary import Summary
import poptorch

dataset_summary = Summary.from_dataset(dataset)
print(dataset_summary)
max_number_of_nodes = int(dataset_summary.num_nodes.max)
max_number_of_edges = int(dataset_summary.num_edges.max)
print(f"Max number of nodes in the dataset is: {max_number_of_nodes}")
print(f"Max number of edges in the dataset is: {max_number_of_edges}")

batch_size = 128
max_num_nodes_per_batch = max_number_of_nodes * batch_size
max_num_edges_per_batch = max_number_of_edges * batch_size
print(f"{max_num_nodes_per_batch = }")
print(f"{max_num_edges_per_batch = }")

from poptorch_geometric import FixedSizeDataLoader

ipu_dataloader = FixedSizeDataLoader(
    dataset,
    num_nodes=max_num_nodes_per_batch,
    num_edges=max_num_edges_per_batch,
    batch_size=batch_size,
)

sample = next(iter(ipu_dataloader))
print(sample)
print("Shape of y:", sample.y.shape)

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class GcnForBatching(torch.nn.Module):
    def __init__(self, hidden_channels, batch_size):
        super(GcnForBatching, self).__init__()
        torch.manual_seed(1234)
        self.conv = GCNConv(dataset.num_features, hidden_channels, add_self_loops=False)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        self.batch_size = batch_size  # includes the padding graph

    def forward(self, x, edge_index, y, batch):
        # 1. Obtain node embeddings
        x = self.conv(x, edge_index).relu()
        # 2. Pooling layer
        x = global_mean_pool(x, batch, size=self.batch_size)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        if self.training:
            return F.mse_loss(
                x[: (self.batch_size - 1)], y[: (self.batch_size - 1)]
            )  # mask out the null graph from the loss
        return x


model = GcnForBatching(hidden_channels=16, batch_size=batch_size)
optim = poptorch.optim.Adam(model.parameters(), lr=0.001)
poptorch_model = poptorch.trainingModel(
    model,
    optimizer=optim,
    options=poptorch.Options().enableExecutableCaching(executable_cache_dir),
)
poptorch_model

poptorch_model.train()
loss_per_epoch = []

for epoch in range(0, 3):
    total_loss = 0

    for data in ipu_dataloader:
        loss = poptorch_model(data.x, data.edge_index, data.y, data.batch)
        total_loss += loss
        optim.zero_grad()

    loss_this_epoch = total_loss / len(dataset)
    loss_per_epoch.append(loss_this_epoch)
    print("Epoch:", epoch, " Training Loss: ", loss_this_epoch)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(loss_per_epoch)
plt.title("Loss per epoch using the Fixed Sized Dataloader")
plt.xlabel("Epoch")
plt.ylabel("Loss")
