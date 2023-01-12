# Cluster GCN model
Cluster graph convolutional networks for node classification, using cluster sampling, optimised for Graphcore's IPU.

Run our Cluster GCN training on arXiv dataset on Paperspace.
<br>
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/3CHtqfy)

| Framework | domain | Model | Datasets | Tasks| Training| Inference | Reference |
|-------------|-|------|-------|-------|-------|---|---|
| TensorFlow2 | GNNs | CGCN | PPI, arXiv, Reddit, Products, MAG, MAG240M | Node classification | ✅ | ❌ | [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf) |


## Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the MolHIV dataset (See Dataset setup)


## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your environment are available in the [poplar quick start guide](https://docs.graphcore.ai/projects/graphcloud-poplar-quick-start/en/latest/).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the Tensorflow2 and IPU Tensorflow add-ons wheels:
```bash
cd <poplar sdk root dir>
pip3 install tensorflow-2.X.X...<OS_arch>...x86_64.whl
pip3 install ipu_tensorflow_addons-2.X.X...any.whl
```
For the CPU architecture you are running on

4. Install the Keras wheel:
```bash
pip3 install --force-reinstall --no-deps keras-2.X.X...any.whl
```
For further information on Keras on the IPU, see the [documentation](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/keras/keras.html#keras-with-ipus) and the [tutorial](https://github.com/graphcore/tutorials/tree/master/tutorials/tensorflow2/keras).

5. Navigate to this example's root directory

6. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

7. Install [METIS](https://doi.org/10.1137/S1064827595287997):
```bash
sudo apt-get install libmetis-dev=5.1.0.dfsg-5
```


## Dataset setup

This implementation of Cluster-GCN running on the Graphcore IPUs supports the following graph datasets
for node prediction. Each of these datasets have a corresponding configuration file that works well, otherwise
these datasets can be selected in the config by setting `dataset_type`.

### PPI (Protein-protein interactions) dataset <a name='ppi' ></a>

The [PPI dataset](https://paperswithcode.com/dataset/ppi) depicts protein roles in various protein-protein
interaction (PPI) graphs. Each graph in the datasets corresponds to a different human tissue. Positional gene sets are
used, motif gene sets and immunological signatures as features and gene ontology sets as multi-class binary labels
(121 in total). The dataset contains in total 56944 nodes, 818716 edges and node feature size 50. The preprocessed PPI
datasets can be downloaded from [Stanford GraphSAGE](https://snap.stanford.edu/graphsage).

### Reddit dataset <a name='reddit' ></a>

The [Reddit dataset](https://paperswithcode.com/dataset/reddit) is a homogeneous graph dataset from Reddit posts made
in the month of September 2014. The node label in this case is the community that a post belongs to. There are 50
large communities  sampled to build a post-to-post graph, connecting posts if the same user comments on both. This
dataset contains in total 232,965 nodes with node feature size of 602 and 114615892 edges. The preprocessed Reddit
datasets can be downloaded from [Stanford GraphSAGE](https://snap.stanford.edu/graphsage).

### arXiv dataset <a name='arxiv' ></a>

The [ogbn-arxiv dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) is a directed homogeneous graph that
encodes a citation network between computer science papers hosted on arXiv. Each paper has a 128-dimmensional node
feature vector, that encodes the title and abstract, processed with a skip-gram model. Each directed link in the
graph indicates that one paper cites another. The task is to predict the correct topic label for the paper from the
40 main categories. The train portion of the dataset is all papers published until 2017, the papers published in 2018
are the validation set, and papers published in 2019 are the test set. To use the arXiv dataset, simply use the
train_arxiv.json config, the dataset will be downloaded automatically.

### Products dataset <a name='products' ></a>

The [ogbn-products dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) is an undirected homogeneous graph that encodes a
network of products. Each product has a 100-dimmensional node feature vector, that encodes the product descriptions.
Each node in the graph represents a product sold in Amazon, with the edges indicating if products are bought together.
The task is to predict the correct category label for a product from the 47 main categories. The train portion of the
dataset makes up only 8% of the dataset and are based on the products sale ranking. 2% validation and the other 90% used
for test. To use this dataset, simply use the train_products.json config, the dataset will be downloaded automatically.

### MAG dataset <a name='mag' ></a>

The [ogbn-mag dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) is a directed heterogeneous graph that is
a subset of Microsoft Academic Graph (MAG). The node types are papers, authors, institutions and fields of study.
These are connected by four edge types, author affiliated with institution, author writes a paper, paper cites a paper,
and paper has a topic of a field of study. Each paper has a 128-dimmensional node feature vector, that
encodes the title and abstract, similar to ogbn-arxiv. The task is to predict the venue of each paper. The train
portion of the dataset is all papers published until 2017, the papers published in 2018 are the validation
set, and papers published in 2019 are the test set. To use this dataset, simply use the train_mag.json config,
the dataset will be downloaded automatically.

### MAG240M dataset <a name='mag240m' ></a>

The [MAG240M dataset](https://ogb.stanford.edu/docs/lsc/mag240m/) is a directed heterogeneous graph that is
a subset of Microsoft Academic Graph (MAG). It is very similar to the above ogbn-mag dataset but is a much larger
subset. The node types are papers, authors, institutions and fields of study. These are connected by four edge types,
author affiliated with institution, author writes a paper, paper cites a paper, and paper has a topic of a field of
study. Each paper has a 768-dimmensional node feature vector, that encodes the title and abstract, similar to
ogbn-arxiv. Only a subset of the papers are the arXiv papers and these are the ones used for training and validation.
The task is to predict the subject areas of these papers. The train portion of the dataset is all papers published
until 2018, the papers published in 2019 are the validation set, and papers published in 2020 are the test set.

This dataset is considerably larger than the others mentioned above. We use the same approach as the
[DeepMind entry for MAG240M-LSC](https://github.com/deepmind/deepmind-research/blob/master/ogb_lsc/mag/README.md#download-and-pre-process-data),
which uses PCA to reduce the feature size. There a script is provided to pre-process the data as well as a script to
download the pre-processed data directly, which you can download from
[DeepMind’s cloud storage](https://storage.googleapis.com/deepmind-ogb-lsc/mag/data/preprocessed/merged_feat_from_paper_feat_pca_129.npy).
Note that the dataset is licensed under ODC-BY.
The path for the downloaded `.npy` file can be given in the `train_mag240.json` config under the `pca_features_path`
parameter, which is expected relative to the data path.
The other parts of the dataset, for example the edges, nodes and labels, will be downloaded automatically when running
the application. The dataset is around 200Gb so can take some time to download (a few hours to a day). The path of this
can be given in the `train_mag240.json` config under `data_path`, or with the `--data-path` argument in the command
line.
For example, the following configuration will load the data from or download to directory
`/graph-datasets/ogb-lsc-mag240`, and will attempt to load the PCA features from file
`/graph-datasets/ogb-lsc-mag240/mag240m_kddcup2021/merged_feat_from_paper_feat_pca_129.npy`:
```json
{
   "data_path": "/localdata/graph-datasets/ogb-lsc-mag240",
   "pca_features_path": "/mag240m_kddcup2021/merged_feat_from_paper_feat_pca_129.npy"
}
```


## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. The benchmarks are provided in the `benchmarks.yml` file in this example's root directory.

For example:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).
