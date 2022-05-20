# Cluster GCN model

## Table of contents

1. [Introduction](#intro)
2. [Prepare environment](#environment)
3. [Datasets](#datasets)
    1. [PPI](#ppi)
    2. [Reddit](#reddit)
    2. [arXiv](#arxiv)
4. [Training and validation](#training_validation)

## Introduction <a name='intro' ></a>

This repository contains a TensorFlow 2 implementation of the Cluster-GCN algorithm, presented in
[Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf), running on the Graphcore IPU.
The model is a Graph convolutional network (GCN) that performs node classification tasks with cluster sampling approach
to enable large-scale training.
In this repository the supported datasets are: Protein-Protein Interaction (PPI), Reddit, and an
arXiv citation network.

## Prepare environment <a name='environment' ></a>

Create a virtual environment and install the appropriate Graphcore TensorFlow 2.5 wheels from inside the SDK directory:

```shell
virtualenv --python python3.6 .gcn_venv
source .gcn_venv/bin/activate
pip install -r requirements.txt
pip install <path to the TensorFlow-2 wheel from the Poplar SDK>
pip install <path to the ipu_tensorflow_addons wheel for TensorFlow 2 from the Poplar SDK>
```

This application uses [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to cluster the graph. To install run the following steps:

```shell
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -xf metis-5.1.0.tar.gz
cd metis-5.1.0
make config shared=1
make
```

## Datasets <a name='datasets' ></a>

This implementation of Cluster-GCN running on the Graphcore IPUs supports the PPI, Reddit, and arXiv graph datasets
for node prediction.
These datasets can be selected in the config by setting `dataset_type` to one of `["ppi", "reddit", "arxiv"]`.

### PPI (Protein-protein interactions) dataset <a name='ppi' ></a>

The [PPI dataset](https://paperswithcode.com/dataset/ppi) depicts protein roles in various protein-protein 
interaction (PPI) graphs. Each graph in the datasets corresponds to a different human tissue. Positional gene sets are
used, motif gene sets and immunological signatures as features and gene ontology sets as multi-class binary labels 
(121 in total). The dataset contains in total 56944 nodes, 818716 edges and node feature size 50. The preprocessed PPI 
datasets can be downloaded from [here](http://snap.stanford.edu/graphsage).

### Reddit dataset <a name='reddit' ></a>

The [Reddit dataset](https://paperswithcode.com/dataset/reddit) is a graph dataset from Reddit posts made in the month of
September 2014. The node label in this case is the community that a post belongs to. There are 50 large communities 
sampled to build a post-to-post graph, connecting posts if the same user comments on both. This dataset contains in 
total 232,965 nodes with node feature size of 602 and 114615892 edges. The preprocessed Reddit datasets can be 
downloaded from [here](http://snap.stanford.edu/graphsage).

### arXiv dataset <a name='arxiv' ></a>

The [arXiv dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) is a directed graph that encodes a citation 
network between computer science papers hosted on arXiv. Each paper has a 128-dimmensional node feature vector, that 
encodes the title and abstract, processed with a skip-gram model. Each directed link in the graph indicates that one 
paper cites another. The task is to predict the correct topic label for the paper from the 40 main categories. 
The train portion of the dataset is all papers published until 2017, the papers published in 2018 are the validation 
set, and papers published in 2019 are the test set. To use the arXiv dataset, simply use the train_arxiv.json config, the dataset will be downloaded automatically.

## Run training and validation <a name='training_validation' ></a>

```shell
python run_cluster_gcn.py ./configs/train_[dataset].json
```

Depending on the config file chosen, the program trains and validates Cluster-GCN model on the selected dataset
using METIS for clustering. Parameters of the model can be modified via the config file or by using the 
command line arguments.
