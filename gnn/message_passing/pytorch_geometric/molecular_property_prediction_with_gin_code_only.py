# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os
import os.path as osp

import poptorch

poptorch.setLogLevel("ERR")
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-gin"
dataset_directory = os.getenv("DATASET_DIR", "data")

from torch_geometric.datasets import TUDataset

tudataset_root = osp.join(dataset_directory, "TUDataset")
dataset = TUDataset(root=tudataset_root, name="NCI1")

data = dataset[0]
print(f"Graph zero summary:")
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Has self-loops: {data.has_self_loops()}")
print(f"Has edge features: {data.edge_attr is not None}")
print(f"Is undirected: {data.is_undirected()}")
print(f"Molecule label: {data.y.item()}")

dataset_summary = dataset.get_summary()
print(dataset_summary)

split = int(len(dataset) * 0.8)
train_dataset = dataset[split:]
test_dataset = dataset[:split]

num_graphs = 64
total_num_nodes = int(dataset_summary.num_nodes.max * num_graphs)
total_num_edges = int(dataset_summary.num_edges.max * num_graphs)
print(f"total_num_nodes = {total_num_nodes}")
print(f"total_num_edges = {total_num_edges}")

from poptorch_geometric.dataloader import FixedSizeDataLoader

train_loader = FixedSizeDataLoader(
    dataset=train_dataset,
    batch_size=num_graphs,
    num_nodes=total_num_nodes,
    num_edges=total_num_edges,
    collater_args=dict(add_masks_to_batch=True),
    drop_last=True,
)

train_loader_iter = iter(train_loader)
first_sample = next(train_loader_iter)
second_sample = next(train_loader_iter)
print(f"first_sample = {first_sample}")
print(f"second_sample = {second_sample}")

first_sample.nodes_mask

float(first_sample.nodes_mask.sum() / len(first_sample.nodes_mask))

max_num_graphs_per_batch = 300

train_loader = FixedSizeDataLoader(
    dataset=train_dataset,
    batch_size=max_num_graphs_per_batch,
    num_nodes=total_num_nodes,
    num_edges=total_num_edges,
    collater_args=dict(add_masks_to_batch=True),
    drop_last=True,
)
first_sample = next(iter(train_loader))
float(first_sample.nodes_mask.sum() / len(first_sample.nodes_mask))

for i, batch in enumerate(train_loader):
    print(f"Batch {i} has {batch.graphs_mask.sum()} real graphs")

poptorch_options = poptorch.Options()
poptorch_options.deviceIterations(2)
poptorch_options.outputMode(poptorch.OutputMode.All)
poptorch_options.enableExecutableCaching(executable_cache_dir)

train_loader = FixedSizeDataLoader(
    dataset=train_dataset,
    batch_size=max_num_graphs_per_batch,
    num_nodes=total_num_nodes,
    num_edges=total_num_edges,
    collater_args=dict(add_masks_to_batch=True),
    drop_last=True,
    options=poptorch_options,
)

next(iter(train_loader))

from model import GIN

in_channels = dataset.num_node_features
hidden_channels = 32
# output hidden units dimension, one unit for each class
out_channels = 2
# number of GinConv layers
num_conv_layers = 4
# number of hidden layers used in the mlp for each GinConv layer
num_mlp_layers = 2

learning_rate = 1e-2
num_epochs = 200

model = GIN(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_conv_layers=num_conv_layers,
    num_mlp_layers=num_mlp_layers,
    batch_size=max_num_graphs_per_batch,
)

model.train()
model

optimizer = poptorch.optim.Adam(model.parameters(), lr=learning_rate)

poptorch_training_model = poptorch.trainingModel(model, optimizer=optimizer, options=poptorch_options)

from tqdm import tqdm

epoch_bar = tqdm(range(num_epochs))

epoch_losses = []
for epoch in epoch_bar:
    epoch_loss = 0
    for data in train_loader:
        preds, micro_batch_loss = poptorch_training_model(
            data.x,
            data.edge_index,
            data.batch,
            graphs_mask=data.graphs_mask,
            target=data.y,
        )
        epoch_loss += micro_batch_loss.mean()

    # decay learning rate every 50 epochs
    if (epoch + 1) % 50 == 0:
        learning_rate *= 0.5
        optimizer = poptorch.optim.Adam(model.parameters(), lr=learning_rate)
        poptorch_training_model.setOptimizer(optimizer)

    epoch_bar.set_description(f"Epoch {epoch} training loss: {epoch_loss:0.6f}")

    epoch_losses.append(epoch_loss)

poptorch_training_model.detachFromDevice()

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.plot(list(range(num_epochs)), epoch_losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.grid(True)

test_loader = FixedSizeDataLoader(
    dataset=train_dataset,
    batch_size=max_num_graphs_per_batch,
    num_nodes=total_num_nodes,
    num_edges=total_num_edges,
    collater_args=dict(add_masks_to_batch=True),
    drop_last=True,
    options=poptorch_options,
)

model.eval()
poptorch_inference_model = poptorch.inferenceModel(model, options=poptorch_options)

total_correct = 0
num_samples = 0
for data in test_loader:
    scores = poptorch_inference_model(data.x, data.edge_index, data.batch)
    preds = scores.argmax(dim=1)
    total_correct += ((preds.flatten() == data.y.flatten()) * data.graphs_mask.flatten()).sum()
    num_samples += data.graphs_mask.sum()

accuracy = total_correct / num_samples
print(f"Test accuracy: {accuracy}")

poptorch_inference_model.detachFromDevice()
