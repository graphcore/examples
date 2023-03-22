# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os
import os.path as osp

import torch
import poptorch
import pandas as pd
import py3Dmol

from periodictable import elements
from poptorch_geometric.dataloader import CustomFixedSizeDataLoader
from torch_geometric.datasets import QM9
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_fixed_size
from torch_geometric.nn.models import SchNet
from torch_geometric.transforms import Pad
from torch_geometric.transforms.pad import AttrNamePadding
from tqdm import tqdm

from utils import TrainingModule, KNNInteractionGraph, prepare_data, optimize_popart

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

poptorch.setLogLevel("ERR")
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/pyg-schnet"
dataset_directory = os.getenv("DATASET_DIR", "data")
num_ipus = os.getenv("NUM_AVAILABLE_IPU", "4")

qm9_root = osp.join(dataset_directory, "qm9")
dataset = QM9(qm9_root)

len(dataset)

datum = dataset[123244]
datum, datum.z, datum.pos, datum.y[:, 4]

dataset.transform = prepare_data
dataset[123244]

num_atoms = int(datum.z.numel())
xyz = f"{num_atoms}\n\n"

for i in range(num_atoms):
    sym = elements[datum.z[i].item()].symbol
    r = datum.pos[i, :].tolist()
    line = [sym] + [f"{i: 0.08f}" for i in r]
    line = "\t".join(line)
    xyz += f"{line}\n"

view = py3Dmol.view(data=xyz, style={"stick": {}})
view.spin()

num_mols = len(dataset)
num_atoms = []
hl_gap = []

for mol in tqdm(dataset):
    num_atoms.append(mol.z.numel())
    hl_gap.append(float(mol.y))

df = pd.DataFrame({"Number of atoms": num_atoms, "HOMO-LUMO Gap (eV)": hl_gap})
df.describe()

h = plt.figure(figsize=[10, 4])
sns.histplot(data=df, x=df.columns[0], ax=plt.subplot(1, 2, 1), discrete=True)
sns.kdeplot(data=df, x=df.columns[1], ax=plt.subplot(1, 2, 2))
h.show()

loader = DataLoader(dataset, batch_size=4)

it = iter(loader)
next(it), next(it)

batch = Batch.from_data_list([dataset[0]])
batch

torch.manual_seed(0)
cutoff = 10.0
model = SchNet(cutoff=cutoff)
model.eval()
cpu = model(batch.z, batch.pos, batch.batch)
cpu

torch.manual_seed(0)
knn_graph = KNNInteractionGraph(cutoff=cutoff, k=batch.num_nodes - 1)
model = SchNet(cutoff=cutoff, interaction_graph=knn_graph)
model = to_fixed_size(model, batch_size=1)
options = poptorch.Options()
options.enableExecutableCaching(executable_cache_dir)
pop_model = poptorch.inferenceModel(model, options)
ipu = pop_model(batch.z, batch.pos, batch.batch)

ipu

torch.allclose(cpu, ipu)

data = Batch.from_data_list([dataset[0]])
pad_transform = Pad(32, node_pad_value=AttrNamePadding({"z": 0, "pos": 0, "batch": 1}))
padded_batch = pad_transform(data)
padded_batch

torch.manual_seed(0)
model = SchNet(cutoff=cutoff)
model.eval()
padded_cpu = model(padded_batch.z, padded_batch.pos, padded_batch.batch)
padded_cpu

torch.allclose(cpu, padded_cpu[0])

torch.manual_seed(0)
knn_graph = KNNInteractionGraph(cutoff=cutoff, k=batch.num_nodes - 1)
model = SchNet(cutoff=cutoff, interaction_graph=knn_graph)
model = to_fixed_size(model, batch_size=2)
pop_model = poptorch.inferenceModel(model, options)
padded_ipu = pop_model(batch.z, batch.pos, batch.batch)

padded_ipu

torch.allclose(ipu, padded_ipu[0])

pop_model.detachFromDevice()

batch_size = 8

dataloader = CustomFixedSizeDataLoader(dataset, batch_size=batch_size, num_nodes=32 * (batch_size - 1))

dataloader_iter = iter(dataloader)
first_batch = next(dataloader_iter)
second_batch = next(dataloader_iter)
print(first_batch)
print(second_batch)

first_batch.batch

num_test = 10000
num_val = 10000
torch.manual_seed(0)
dataset = dataset.shuffle()
test_dataset = dataset[:num_test]
val_dataset = dataset[num_test : num_test + num_val]
train_dataset = dataset[num_test + num_val :]

print(
    f"Number of test molecules: {len(test_dataset)}\n"
    f"Number of validation molecules: {len(val_dataset)}\n"
    f"Number of training molecules: {len(train_dataset)}"
)

replication_factor = int(num_ipus)
device_iterations = 32
gradient_accumulation = max(1, 16 // replication_factor)
learning_rate = 1e-4
num_epochs = 5

options = poptorch.Options()
options.enableExecutableCaching(executable_cache_dir)
options.outputMode(poptorch.OutputMode.All)
options.deviceIterations(device_iterations)
options.replicationFactor(replication_factor)
options.Training.gradientAccumulation(gradient_accumulation)

additional_optimizations = True

if additional_optimizations:
    options = optimize_popart(options)

train_loader = CustomFixedSizeDataLoader(
    train_dataset,
    batch_size=batch_size,
    num_nodes=32 * (batch_size - 1),
    options=options,
)

print(next(iter(train_loader)))

torch.manual_seed(0)
knn_graph = KNNInteractionGraph(cutoff=cutoff, k=28)
model = SchNet(cutoff=cutoff, interaction_graph=knn_graph)
model.train()
model = TrainingModule(model, batch_size=batch_size, replace_softplus=additional_optimizations)
optimizer = poptorch.optim.AdamW(model.parameters(), lr=learning_rate)
training_model = poptorch.trainingModel(model, options, optimizer)

data = next(iter(train_loader))
training_model.compile(data.z, data.pos, data.batch, data.y)

train = []

for epoch in range(num_epochs):
    bar = tqdm(train_loader)
    for i, data in enumerate(bar):
        _, mini_batch_loss = training_model(data.z, data.pos, data.batch, data.y)
        loss = float(mini_batch_loss.mean())
        train.append({"epoch": epoch, "step": i, "loss": loss})
        bar.set_description(f"Epoch {epoch} loss: {loss:0.6f}")

training_model.detachFromDevice()

df = pd.DataFrame(train)
g = sns.lineplot(data=df[df.epoch > 0], x="epoch", y="loss", errorbar="sd")
g.set_xticks(range(0, num_epochs + 2, 2))
g.figure.show()
