# Sampling Large Graphs for IPUs using PyTorch Geometric





In the previous PyTorch Geometric (PyG) tutorials we have been focussed on working with datasets comprising of many small graphs. For some modern applications, however, we will need to operate on larger graphs characterised by increasing number of nodes (range 10M-10B) and edges (range 100M-100B). Imagine having to build a recommendation system for a social network type of input graph, which can contain a huge number of users (nodes) and relationships (edges).

There are two ways of approaching large graph problems:
- Full batch training: This is the approach we have been using in the tutorial [An End-to-end Example using PyTorch Geometric on IPUs](https://ipu.dev/O2u8MZ) when working with a single, relatively small graph. The aim is to generate embeddings for all the nodes at the same time. This means keeping the entire graph, as well as all the node embeddings, in memory. If the size of the computational graph increases, the amount of memory required to hold graph and embeddings becomes challenging for modern accelerators.
- Mini-batching: We can also create mini-batches by sampling from the graph. When sampling from a larger graph we need to be extra careful to reduce the chances of the sampled nodes being isolated from each other. If the sampled nodes are isolated, then the mini-batches would no longer be representative of the whole graph. This would negatively impact our machine learning task. The need here is to engineer effective sampling methods to make sure that the message passing scheme is still effective with large graphs.

In this tutorial, we will demonstrate two sampling approaches widely used in literature to cope with increasing graph size. For each we will finish with training using the Graphcore IPU, which is a great fit for GNN applications as explained in the blog post [Accelerating PyG on IPUs: Unleash the Power of Graph Neural Networks](https://www.graphcore.ai/posts/accelerating-pyg-on-ipus-unleash-the-power-of-graph-neural-networks), and the PyG integration for IPUs. You will learn how to:
- effectively cluster nodes of your input graph;
- sample neighbouring nodes of your input graph;
- then, for both sampling methods, train your GNN on IPUs to classify papers from the PubMed dataset.

This notebook assumes some familiarity with PopTorch, PyG and PopTorch Geometric. If you need more information, please consult:
- [PopTorch documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html),
- [PopTorch examples and tutorials](https://docs.graphcore.ai/en/latest/examples.html#pytorch),
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/),
- [PopTorch Geometric documentation](https://docs.graphcore.ai/projects/poptorch-geometric-user-guide/en/latest/index.html).

[![Join our Slack
Community](https://img.shields.io/badge/Slack-Join%20Graphcore's%20Community-blue?style=flat-square&logo=slack)](https://www.graphcore.ai/join-community)

## Environment setup

The best way to run this demo is on Paperspace Gradient's cloud IPUs because everything is already set up for you. To run the demo using other IPU hardware, you need to have the Poplar SDK enabled and the latest PopTorch Geometric wheel installed. Refer to the [getting started guide](https://docs.graphcore.ai/en/latest/getting-started.html#getting-started) for your system for details on how to enable the Poplar SDK and install the PopTorch wheels.

Install the dependencies the notebook needs.

```shell
pip install -r ../requirements.txt
```

To make it easier for you to run this tutorial, we read in some configuration related to the environment you are running the notebook in.

```python
import os

dataset_directory = os.getenv("DATASETS_DIR", "data")
```

## Loading a large graph

We will use the `PubMed` dataset from the `Planetoid` node classification benchmarking suite to demonstrate both the clustering and the neighborhood sampling methods. Let's load it and print some statistics about it.

```python
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
```

As we can see above, the dataset consists of 19,717 scientific publications from the PubMed database relative to diabetes classified into one of three classes. This is not a huge dataset, but it will serve our purpose to demonstrate the sampling approaches.

## Clustering to train the large graph for node classification

The idea behind this method is that we can split the entire input graph into smaller sub-graph clusters. When loading the data into the model, we combine a number of clusters together, regrowing the edges removed in the clustering process. This graph of combined clusters is the mini-batch on which we can calculate layer-wise embeddings by performing message passing.
The clusters should also retain the connectivity of the original graph to avoid information loss: to achieve that we make sure that the small communities from the original graph are mirrored in the generated clusters, by minimising the amount of edge cuts when forming the clusters

### Clustering the graph

A well known approach is [Cluster-GCN](https://arxiv.org/abs/1905.07953). The steps are:
- pre-processing: given a large graph, we partition it into groups of nodes and edges we name clusters. The clusters are formed by minimising the edge cuts required to achieve clusters of a particular number of nodes.
- mini-batch training: we load multiple clusters, re-instate the edges removed by the pre-processing clustering step, and load this reconstructed sub-graph in the device memory and apply message passing over it to compute the loss.

You can also check out our [Cluster-GCN example.](https://ipu.dev/PmAtSw)

Let's now proceed with the clustering. The first step is to use `torch_geometric.data.ClusterData` to partition our `Data` object into `num_clusters` clusters. Under the hood it leverages the METIS algorithm to obtain the splits.

```python
from torch_geometric.loader import ClusterData

num_clusters = 100

cluster_data = ClusterData(
    data, num_parts=num_clusters, recursive=False, save_dir=dataset_directory
)

print(f"The dataset has been split in {len(cluster_data)} clusters")
```

We can now use the PyG `torch_geometric.loader.ClusterLoader` class which merges clusters and their between-clusters links into mini-batches.

```python
from torch_geometric.loader import ClusterLoader

clusters_per_batch = 10

dynamic_size_dataloader = ClusterLoader(
    cluster_data,
    batch_size=clusters_per_batch,
)

dynamic_dataloader_iter = iter(dynamic_size_dataloader)
print(f"{next(dynamic_dataloader_iter) = }")
print(f"{next(dynamic_dataloader_iter) = }")
```

 The mini-batches have been created by combining clusters together, hence each mini-batch has a different size. You can see how each mini-batch contains a different number of nodes and edges in the cell above. We will need to keep this in mind when preparing the mini-batches to feed into the IPU. The IPU relies on ahead-of-time (AOT) compilation, hence it needs the input tensors to have fixed sizes.  Therefore we need to make sure that the mini-batches produced are fixed in size so they can be loaded correctly onto the IPU.

Now we need an approach to make the mini-batches of clusters a fixed size. We will use the `poptorch_geometric.FixedSizeClusterLoader` for this purpose. To use this loader, we need to find the maximum number of nodes and edges in the mini-batches. The `FixedSizeClusterLoader` will then pad the mini-batches accordingly. To do so, we can use the helper method `FixedSizeOptions.from_loader` which handles this calculation by sampling the previously defined dynamic data loader, returning a `FixedSizeOption` object specifying the number of nodes and edges to pad the mini-batches to.

```python
import poptorch
from poptorch_geometric import FixedSizeOptions, OverSizeStrategy

fixed_size_options = FixedSizeOptions.from_loader(
    dynamic_size_dataloader, sample_limit=10
)

print(fixed_size_options)
```

As we can see in the cell above, we now have a maximum number of nodes and edges that the `FixedSizeClusterLoader` loader from PopTorch Geometric will use to pad the mini-batches to. Let's see how:

```python
from poptorch_geometric.cluster_loader import FixedSizeClusterLoader

train_dataloader = FixedSizeClusterLoader(
    cluster_data,
    batch_size=clusters_per_batch,
    fixed_size_options=fixed_size_options,
    over_size_strategy=OverSizeStrategy.TrimNodesAndEdges,
)
```

Let's now inspect a couple of mini-batches loaded by the dataloader:

```python
train_dataloader_iter = iter(train_dataloader)
print(f"{next(train_dataloader_iter) = }")
print(f"{next(train_dataloader_iter) = }")
```

As expected, the loader has used the maximum number of nodes and edges contained in the `FixedSizeOptions` object and all the tensor shapes are consistent across mini-batches. Now that we have clustered and loaded the data to be compatible with the AOT requirement of the IPU, the next step is to train a GNN model to classify the papers in our dataset.

### Training a GNN to classify papers in the PubMed dataset

The first step is to define a GNN model to carry out our classification task. We can easily re-use one of the models we defined in another tutorial, for example a simple GCN-based model, as using clustering does not impact model definition:

```python
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
```

We can now create the PopTorch model and train it on the IPU:

```python
model = GCN(dataset.num_features, dataset.num_classes)
model

from torchinfo import summary

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
poptorch_model = poptorch.trainingModel(model, optimizer=optimizer)

summary(poptorch_model)
```

The training loop looks like this:

```python
from tqdm import tqdm

num_epochs = 5
train_losses = torch.empty(num_epochs, len(train_dataloader))

for epoch in range(num_epochs):
    bar = tqdm(train_dataloader)
    for i, batch in enumerate(bar):
        _, mini_batch_loss = poptorch_model(
            batch.x, batch.edge_index, batch.train_mask, batch.y
        )
        train_losses[epoch, i] = float(mini_batch_loss.mean())
        bar.set_description(
            f"Epoch {epoch} training loss: {train_losses[epoch, i].item():0.6f}"
        )
        optimizer.zero_grad()  # clear gradients
```

We can now detach the training model from the IPU:

```python
poptorch_model.detachFromDevice()
```

Finally, we can plot the mean of the loss to verify that it decreases nicely:

```python
import matplotlib.pyplot as plt

plt.plot(train_losses.mean(dim=1))
plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.legend(["Mean training loss per epoch"])
plt.xticks(torch.arange(0, num_epochs, 2))
plt.gcf().set_dpi(150)
```

## Neighbourhood sampling the computation graph for node classification

The neighbourhood sampling approach was firstly introduced in the [GraphSAGE](https://arxiv.org/abs/1706.02216v4) paper for creating inductive node embeddings. The idea here is to learn how to aggregate node feature information from a node's K-hop neighbourhood, where K is the number of layers of our GNN and the number of iterations we want to perform. This means we have relatively small memory requirements. To generate the embeddings of a node we only need to know the K-hop neighbourhood structure around that node and the relative node features. We don't need to store the rest of the graph. In this way, the mini-batches are going to be made of K-hop neighbourhoods which should fit nicely into the device's memory.

### Neighbour sampling the graph

PyG provides the `torch_geometric.loader.NeighborLoader`, a data loader that performs neighbour sampling as per the GraphSAGE paper, allowing for mini-batching when a full-batch update is not feasible. We can select how many neighbouring nodes we'd like to be sampled for each node in each iteration, as well as the number of iterations (or number of hops) we'd like to go through. In this case, the `batch_size` argument represents how many nodes we want to sample from, which we can agree to refer to as 'target' nodes. Let's take a look at how to implement this in code:

```python
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
print(
    f"Original graph node index of each target node in the mini-batch: {sampled_data.input_id}"
)
print(
    f"Original graph node index of each node in the mini-batch: {sampled_data.n_id}"
)  # shows the target nodes are the first 5
```

In PyG, sampled nodes are sorted based on the order they were sampled. From inspecting the elements in the iterator, we can see how the first `batch_size` nodes (5 in this case) in the sampled sub-graph are the target nodes. In other words, they represent the set of original mini-batch nodes and we will use these nodes for training later on.

To comply with the AOT requirement in PopTorch on IPUs, we need to change the variable tensor sizes shown in the above cell and make them fixed size. To this end, we will leverage the `poptorch_geometric.FixedSizeNeighborLoader` in PopTorch Geometric. Under the hood, this class takes the sub-graphs generated by the PyG `NeighborLoader` class and then applies the PopTorch Geometric `FixedSizeCollater` class to make sizes of the resulting shapes of the sub-graphs fixed size across sub-graphs.

Similarly to what we did in the clustering demonstration, we will first use the helper method `FixedSizeOptions.from_loader` which returns a `FixedSizeOption` object specifying the number of nodes and edges to pad the mini-batches to.

```python
fixed_size_options = FixedSizeOptions.from_loader(
    train_loader_sampling, sample_limit=10
)

print(fixed_size_options)
```

We can pass the obtained maximum number of nodes and edges to `poptorch_geometric.FixedSizeNeighborLoader`:

```python
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
```

As we can see, the resulting sub-graphs and tensor shapes have now fixed sizes using the number of nodes and edges previously determined!

### Training using neighborhood sampling on IPUs

We will use a slight modification of the model we defined for the clustering approach demonstration and repeat the same steps to carry out our node classification task. We will leverage the `SAGEConv()` PyG layer to perform one iteration of the aggregate-and-update step. The selected aggregation scheme is the default one, a mean. We have to be careful about selecting which nodes we actually want to consider for training in each sub-graph. We need to make sure we train on those nodes we defined as the target nodes and that we leave the padding sub-graph in each sub-graph out of the loss calculation.

```python
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
        _, mini_batch_loss = poptorch_model_sampling(
            batch.x, batch.edge_index, batch.train_mask, batch.y
        )
        epoch_losses[epoch, i] = float(mini_batch_loss.mean())
        bar.set_description(
            f"Epoch {epoch} training loss: {epoch_losses[epoch, i].item():0.6f}"
        )

import matplotlib.pyplot as plt

plt.plot(epoch_losses.mean(dim=1))
plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.legend(["Mean training loss per epoch"])
plt.grid(True)
plt.xticks(torch.arange(0, num_epochs, 2))
plt.gcf().set_dpi(150)
```

## Conclusion

In this tutorial, we explored the main methods to deal with large graphs that otherwise wouldn't fit in memory, using two different sampling approaches and dedicated dataloaders to optimise performance on Graphcore IPUs.

We demonstrated:
- how to effectively cluster nodes of your input graph using `FixedSizeClusterLoader` while the AOT requirements on the IPU;
- how to sample neighbouring nodes of your input graph, using `poptorch_geometric.FixedSizeNeighborLoader`;
- how to train your GNN on IPUs to classify papers from the PubMed dataset.

In this tutorial we have worked with homogeneous graphs, however scaled up GNN problems are also very well suited to heterogeneous graphs: please check out our [heterogeneous graphs tutorial](../6_heterogeneous_graphs) to know more about how to handle those on IPUs. More extensive node classification examples can be found in our [Gradient-Pytorch-Geometric/node-prediction](https://github.com/graphcore/Gradient-Pytorch-Geometric/tree/main/node-prediction) repository.
