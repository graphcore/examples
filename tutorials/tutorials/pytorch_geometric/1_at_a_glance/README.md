# PyTorch Geometric on IPUs at a glance

IPUs can significantly accelerate both training and inference on GNNs. To use an existing PyTorch Geometric (PyG) model on IPUs some minor changes are needed. Some of these changes are required so that the model can run on IPUs, and other changes are optional for improving performance.

In this tutorial you will learn how to:

- Run an existing PyTorch Geometric model on the IPU,
- Accelerate your dataloader performance using the PopTorch (IPU-specific set of extensions for PyTorch) dataloader, while satisfying the static graph requirements of the IPU by using fixed sized inputs,
- Make the necessary changes in some PyTorch Geometric layers and operations to meet the static graph requirements of the IPU.

While this tutorial will cover enough of the basics of GNNs, PyTorch Geometric and PopTorch
for you to start developing and porting your GNN applications to the IPU;
the following resources can be used to complement your understanding of:

- PopTorch : [Introduction to PopTorch - running a simple model](https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch/basics);
- GNNs : [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- PyTorch Geometric (PyG): [Official notebooks examples and tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)

## Running on Paperspace

The Paperspace environment lets you run this notebook with no set up. To improve your experience we preload datasets and pre-install packages, this can take a few minutes, if you experience errors immediately after starting a session please try restarting the kernel before contacting support. If a problem persists or you want to give us feedback on the content of this notebook, please reach out to through our community of developers using our [slack channel](https://www.graphcore.ai/join-community) or raise a [GitHub issue](https://github.com/graphcore/examples).

Requirements:

* Python packages installed with `pip install -r ../requirements.txt`

```bash
pip install -r ../requirements.txt
```

And for compatibility with the Paperspace environment variables we will do the following:

```python
import os

dataset_directory = os.getenv("DATASET_DIR", "data")
```

Now we are ready to start!

## Porting to the IPU basics

To run your model using PyTorch Geometric on the IPU, the model will need to target PopTorch. PopTorch is a set of IPU-specific extensions which allows you to run PyTorch native models on the IPU.
It is designed to require as few changes as possible from native PyTorch, but there are some differences. This means a few changes are required:

* Move the loss function inside the `forward` method of your model.
* Wrap the model in `poptorch.trainingModel` or `poptorch.inferenceModel`.
* Remove the manual call to the backward pass and optimizer steps - both are handled by PopTorch automatically.

Additional useful changes to make:
* Use a PopTorch optimizer, specifically designed for the IPU.

Let's see what these changes mean by taking a look at a small example.
First let's load a dataset: the Cora dataset is a citation network where a node represents a document and an edge exists if there is a citation between the two documents.

```python
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

dataset = Planetoid(dataset_directory, "Cora", transform=T.NormalizeFeatures())
data = dataset[0]
print(data)
```

```output
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
```

Let's look at a typical training example. We will use a GCN layer, one of the most commonly used GNN operators.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234)
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv(x, edge_index, edge_weight).relu()
        return x


model = GCN(dataset.num_features, dataset.num_classes)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training on CPU.")

for epoch in range(1, 6):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss}")
```

```output
Training on CPU.
Epoch: 1, Loss: 1.9448680877685547
Epoch: 2, Loss: 1.9443432092666626
Epoch: 3, Loss: 1.9438152313232422
Epoch: 4, Loss: 1.9431666135787964
Epoch: 5, Loss: 1.9429877996444702
```

Now let's make the changes mentioned above to make this example run on the IPU.

```python
import poptorch


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234)
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, y, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv(x, edge_index, edge_weight).relu()

        if self.training:
            loss = F.cross_entropy(x, y)
            return x, loss
        return x


model = GCN(dataset.num_features, dataset.num_classes)
model.train()
optimizer = poptorch.optim.Adam(model.parameters(), lr=0.001)
poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)

print("Training on IPU.")
for epoch in range(1, 6):
    output, loss = poptorch_model(
        data.x, data.edge_index, data.y, edge_weight=data.edge_attr
    )
    print(f"Epoch: {epoch}, Loss: {loss}")
```

```output
Training on IPU.
[10:21:41.138] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 5
[10:21:41.138] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 6
[10:21:41.146] [poptorch:cpp] [warning] %86 : float = prim::Constant() # /tmp/ipykernel_19919/3481811172.py:12:0: torch.float64 constant cannot be represented as a torch.float32
Graph compilation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:31<00:00]
Epoch: 1, Loss: 1.944833517074585
Epoch: 2, Loss: 1.944195032119751
Epoch: 3, Loss: 1.9437580108642578
Epoch: 4, Loss: 1.9431732892990112
Epoch: 5, Loss: 1.9426839351654053
```

You have now successfully compiled and run the model on IPU!

We have seen the changes required to get training your PyTorch Geometric model on IPU. For more comprehensive information please refer to the [PopTorch documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html).

Now let's take a look at some of the other changes that are useful to get more performance out of the IPU.

## High performance dataloader and fixed size inputs

PopTorch provides its own dataloader that behaves very similarly to the PyTorch dataloader you may be familiar with, `torch.utils.data.DataLoader`. The [PopTorch dataloader](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/pytorch_to_poptorch.html#preparing-your-data) provides the following features:

* It takes a `poptorch.Options` instance to use IPU-specific features for example [deviceIterations](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html?highlight=deviceIterations#poptorch-options-deviceiterations);
* It automatically computes the number of elements consumed by a single step;
* It enables asynchronous data loading.

[PopTorch Geometric](https://docs.graphcore.ai/projects/poptorch-geometric-user-guide/), the IPU-specific PyTorch Geometric library, provides a wrapper for the PopTorch dataloader, making it easy to get performant PyTorch Geometric models running on the IPU. Let's see how to get started with it.

First we load a dataset. In this case we are loading the MUTAG dataset, which is a collection of many small graphs>

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(dataset_directory, name="MUTAG")
data = dataset[0]
print(data)
```

```output
Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])
```

To create a dataloader in PyTorch Geometric we do the following:

```python
from torch_geometric.loader import DataLoader

torch.manual_seed(1234)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```

The IPU needs fixed sized inputs, which means that prior knowledge of the shape of the input tensors is required.
There are different ways to achieve fixed sized inputs, and the method used will depend on the type of input graph dataset we're working with:
* if we're dealing with a dataset of many small graphs, we can batch the input graphs via the dataloader and pad the resulting batch: you can check out our tutorial on [Small Graph Batching with Padding](../3_small_graph_batching_with_padding/3_small_graph_batching_with_padding.ipynb) for a detailed walkthrough. This approach may result in a very large amount of padding in specific use cases: we present a more efficient batching strategy called packing in a dedicated tutorial on [Small Graph Batching with Packing](../4_small_graph_batching_with_packing/4_small_graph_batching_with_packing.ipynb).
* if we're dealing with a dataset of a single large graph, we can sample from it and then pad the samples to obtain static shapes. You can refer to the [Cluster CGN example](../../../../gnn/cluster_gcn/pytorch_geometric/node_classification_with_cluster_gcn.ipynb) for a large graph use case.

We demonstrate the usage of `FixedSizeDataLoader`, a class to create a fixed batch sampler with `batch_size` graphs in each batch.
The `num_nodes` and `num_edges` optional arguments allow you to set the total number of nodes and edges in a batch, respectively, to make the batch fixed size and therefore suitable for the IPU.
We can inspect the dataset using the `Summary` helper functionality to collect some statistics on the number of nodes and edges in the dataset: this will help us decide which `num_nodes` and `num_edges` to use in the dataloader.

```python
from poptorch_geometric import FixedSizeDataLoader
from torch_geometric.data.summary import Summary

torch.manual_seed(1234)

dataset_summary = Summary.from_dataset(dataset)
dataset_summary
max_number_of_nodes = int(dataset_summary.num_nodes.max)
max_number_of_edges = int(dataset_summary.num_edges.max)
print(f"Max number of nodes in the dataset is: {max_number_of_nodes}")
print(f"Max number of edges in the dataset is: {max_number_of_edges}")

ipu_dataloader = FixedSizeDataLoader(
    dataset, num_nodes=300, num_edges=600, batch_size=10
)
```

```output
[10:22:14.731] [poptorch::python] [warning] The `batch_sampler` __len__ method is not implemented and drop_last=False. The last tensor may be incomplete - batch size < 1. To avoid having to handle this special case switch to drop_last=True.
Max number of nodes in the dataset is: 28
Max number of edges in the dataset is: 66
```

If we look at the what the dataloader has produced, you will see that `ipu_dataloader` produces `batch_size` mini-batches with the specified number of nodes and edge to work with fixed size inputs.
The other dimensions match the PyTorch Geometric dataloader.

```python
print(f"{next(iter(dataloader)) = }")
print(f"{next(iter(ipu_dataloader)) = }")
```

```output
next(iter(dataloader)) = DataBatch(edge_index=[2, 404], x=[184, 7], edge_attr=[404, 4], y=[10], batch=[184], ptr=[11])
next(iter(ipu_dataloader)) = DataBatch(x=[300, 7], edge_index=[2, 600], edge_attr=[600, 4], y=[10], batch=[300], ptr=[11], num_nodes=300, num_edges=600)
```

Let's define our GCN based model.

```python
from torch_geometric.nn import global_mean_pool


class GcnForIpu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch_size):
        super().__init__()
        torch.manual_seed(1234)
        self.batch_size = batch_size
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, y, batch):
        x = self.conv(x, edge_index).relu()

        x = global_mean_pool(x, batch, size=self.batch_size)

        if self.training:
            loss = F.cross_entropy(x, y)
            return x, loss

        return x
```

Now we can use the dataloader with our model.

```python
model = GcnForIpu(dataset.num_features, dataset.num_classes, batch_size=10)

optim = poptorch.optim.Adam(model.parameters(), lr=0.01)
poptorch_model = poptorch.trainingModel(model, optimizer=optim)
poptorch_model.train()

in_data = next(iter(ipu_dataloader))
poptorch_model(in_data.x, in_data.edge_index, in_data.y, in_data.batch)
```

```output
[10:22:14.760] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 53
[10:22:14.761] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 54
[10:22:14.761] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 55
[10:22:14.771] [poptorch:cpp] [warning] %91 : float = prim::Constant() # /tmp/ipykernel_19919/3517426405.py:12:0: torch.float64 constant cannot be represented as a torch.float32
Graph compilation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:06<00:00]
```

```output
(tensor([[0.0180, 0.0072],
         [0.0214, 0.0086],
         [0.0286, 0.0115],
         [0.0264, 0.0106],
         [0.0214, 0.0086],
         [0.0311, 0.0125],
         [0.0228, 0.0092],
         [0.0311, 0.0125],
         [0.0264, 0.0106],
         [0.0000, 0.0000]]),
 tensor(0.6935))
```

We can extend this simple example to make use of some of the PopTorch features mentioned above, for example increasing the number of device iterations. This will mean running the training loop on the IPU over that `deviceIterations` number of iterations, preparing this number of mini-batches on the host so these iterations can be done faster.

Using the standard PopTorch dataloader unlocks some more very useful features, see the PopTorch tutorial on [Efficient Data Loading](https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch/efficient_data_loading).

## Operation and layer considerations

There are particular operations and layers that have to be taken into consideration when porting your model to the IPU. Many of these are because of having to compile a static graph for the IPU. Each has a simple solution which we describe below.

### Operations

#### Boolean indexing

Indexing a tensor with a tensor of booleans can result in a tensor that isn't a fixed size in every case. This invalidates the IPU requirement of having a static graph. These operations are used in many places, for example in the calculation of the loss when a mask is applied to the final activations. We can see this in the following operation.

```python
dataset = Planetoid(dataset_directory, "Cora", transform=T.NormalizeFeatures())
data = dataset[0]
```

Typically we would do the following to apply the mask.

```python
x = data.x[data.train_mask]
y = data.y[data.train_mask]
loss = F.cross_entropy(x, y)
```

Depending on the number of true values in `train_mask` then `x` will be a different size per sample and therefore does not fulfill the requirement of a static graph for IPU. To avoid this we can use `torch.where` which will produce a fixed size output.

```python
y = torch.where(data.train_mask, data.y, -100)
loss = F.cross_entropy(data.x, y)
```

Here `y` is a fixed size independent of how many true values are in `train_mask`. Here we also use the fact that `-100` is ignored by default in the loss function, therefore we populate the masked `y` values with `-100` and can skip the masking operation on `x`.

### PyTorch Geometric Layers

A few common layers used in PyTorch Geometric have features that need to be considered when using them with IPUs. These are listed below with solutions.

#### Global pooling layers

Global pooling layers are very common in PyTorch Geometric, for example `global_mean_pool`, `global_max_pool` and `global_add_pool`. These layers attempt to calculate the batch size if not provided which cannot be done automatically on the IPU.

```python
from torch_geometric.nn import global_mean_pool

x = global_mean_pool(data.x, data.batch)
```

Instead can specify the batch size as an input of the pooling function to avoid this automatic calculation.

```python
batch_size = 1
x = global_mean_pool(data.x, data.batch, size=batch_size)
```

#### GCNConv layers

The `GCNConv` layer adds self-loops to the input graph by default. Self-loops are only added to those nodes that don't already have an existing self-loop. This results in the output having an unpredictable size and therefore does not fulfill the requirement that the graph must be static for the IPU. To avoid this we can do the following.

First let's look at the layer, with self-loops turned on.

```python
conv = GCNConv(in_channels=10, out_channels=10)
conv
```

```output
GCNConv(10, 10)
```

We can force this layer to not add the self-loops and instead add them at the dataset loading stage. Let's turn off the self-loops in the layer.

```python
conv = GCNConv(in_channels=10, out_channels=10, add_self_loops=False)
conv
```

```output
GCNConv(10, 10)
```

Then we need to ensure these self-loops exist in the dataset samples. We can use a transform to do this.

```python
import torch_geometric.transforms as T

transform = T.AddSelfLoops()
transform
```

```output
AddSelfLoops()
```

And then apply this transformation to the dataset, for example as a pretransform, shown below.

```python
dataset = TUDataset(
    f"{dataset_directory}/self_loops", name="MUTAG", pre_transform=transform
)
dataset
```

```output
MUTAG(188)
```

Now the data itself contains self-loops and they aren't required to be added in the GCN conv layer.

## Conclusion

In this tutorial, we have discussed the aspects that must be considered when using PyTorch Geometric on IPUs.

You should now have a good understanding of:
* How to port an existing PyTorch Geometric model to run on the IPU.
* How to get the most out of dataloading when using the IPU while respecting the requirement of fixed size inputs.

For the next steps you can explore some of our other [tutorials](..), which look more in depth at some of the topics discussed here.
Or take a look at our GNN examples which dive into more specific applications using state of the art models: for instance, take a look at our [Schnet Notebook](../../../../gnn/schnet/pytorch_geometric/molecular_property_prediction_with_schnet.ipynb).
