# Message Passing Neural Network (MPNN)

## Table of contents

1. [Overview of the Architectures](#overview)
2. [Setup](#setup)
3. [Dataset](#dataset)
    1. [Model Details](#model)
    2. [Performance](#performance)
4. [Benchmarking](#benchmarking)
5. [Profiling](#profiling)
6. [Changelog](#changelog)
7. [Licensing](#licensing)

## Overview of the Architectures <a name='overview' ></a>

In this repository, we use the Graphcore IPU to implement a few popular graph neural network architectures.

For training, `--model==graph_isomorphism` uses a Graph Isomorphism Network (GIN)[1]. `--model==graph_network` uses a Graph Network[2], and `--model=interaction_network` implements an Interaction Network[3]. These are all used to predict chemical properties of molecules on the IPU.

The script `run_training.py` runs and evaluates training. This repository is written in TensorFlow 2 and uses Keras extensively.

This repository supports training, evaluation, and benchmarking the throughput.

<!--
-->
*[1] Xu, Keyulu, et al. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).*

*[2] Battaglia, Peter W., et al. "Relational inductive biases, deep learning, and graph networks." arXiv preprint arXiv:1806.01261 (2018).*

*[3] Gilmer, Justin, et al. "Neural message passing for quantum chemistry." International conference on machine learning. PMLR, 2017.*

## Setup <a name='setup' ></a>

Create a virtual environment and install the appropriate Graphcore TensorFlow 2 wheels from inside
the SDK directory:

```shell
virtualenv --python python3.6 .gcn_venv
source .gcn_venv/bin/activate
pip install -r requirements.txt
pip install <path to the TensorFlow-2 wheel from the Poplar SDK>
pip install --force-reinstall --no-deps <path to the Keras wheel from the Poplar SDK>
pip install <path to the ipu_tensorflow_addons wheel for TensorFlow 2 from the Poplar SDK>
```

Our IPU implementation uses a TensorFlow custom op that has to be compiled prior to use.
First, make sure you have sourced your TensorFlow environment, then from
your top directory (the one containing this README) run
`cd static_ops && make && cd -`.

### Quick-start

To train a Graph Isomorphism Network, run:

```shell
python run_training.py --model=graph_isomorphism --n_graph_layers=5  --nodes_dropout=0.1
```

This should get a test AUC of slightly above 0.75.

## Dataset <a name='dataset' ></a>

This example trains using the Open Graph Benchmark [4] — specifically, the `molhiv` dataset. This contains over 40,000 molecules. The machine learning task is to predict whether a molecule inhibits HIV replication.

The leaderboard for this task is found [here](https://ogb.stanford.edu/docs/leader_graphprop/).

*[4] Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." arXiv preprint arXiv:2005.00687 (2020).*

### Model Details <a name='model' ></a>

To integrate the use of edges (i.e. the information from atomic bonds), we follow [5] and embed the edges afresh at each layer. These embedded edges are combined with the neighborhood aggregation from the nodes.

We used similar parameters to the example in [5], but use Layer Normalization instead of Batch Normalization at the hidden layers of the multi-layer perceptrons.

*[5] Hu, Weihua, et al. "Strategies for pre-training graph neural networks." arXiv preprint arXiv:1905.12265 (2019).*

### Performance <a name='performance' ></a>

We compare our model with the GIN implemented on the [OGB leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv). As per the instructions, training is performed 10 times and the results reported.

The command used is: `python run_training.py --micro_batch_size=24 --cosine_lr=True --epochs=150  --nodes_dropout=.1`, which uses FP16 training.

The test results are consistent with the reported results.

| Model | Parameters | Test ROC-AUC | Validation ROC-AUC |
|----|----|----|----|
| GIN on leaderboard | 1,885,206| 0.7558 ± 0.0140 | 0.8232 ± 0.0090 |
| IPU GIN model | 1,708,106 | 0.7574 ± 0.0215 | 0.7915 ± 0.0114|

## Benchmarking <a name='benchmarking' ></a>

To reproduce the benchmarks, please follow the setup instructions in this README to setup the environment, and then from this dir, use the `examples_utils` module to run one or more benchmarks. For example:

```shell
python3 -m examples_utils benchmark --spec benchmarks.yml
```

or to run a specific benchmark in the `benchmarks.yml` file provided:

```shell
python3 -m examples_utils benchmark --spec benchmarks.yml --benchmark <benchmark_name>
```

For more information on how to use the examples_utils benchmark functionality, please see the <a>benchmarking readme<a href=https://github.com/graphcore/examples-utils/tree/master/examples_utils/benchmarks>

## Profiling <a name='profiling' ></a>

Profiling can be done easily via the `examples_utils` module, simply by adding the `--profile` argument when using the `benchmark` submodule (see the <strong>Benchmarking</strong> section above for further details on use). For example:

```shell
python3 -m examples_utils benchmark --spec benchmarks.yml --profile
```

Will create folders containing popvision profiles in this applications root directory (where the benchmark has to be run from), each folder ending with "_profile". 

The `--profile` argument works by allowing the `examples_utils` module to update the `POPLAR_ENGINE_OPTIONS` environment variable in the environment the benchmark is being run in, by setting:

```
POPLAR_ENGINE_OPTIONS = {
    "autoReport.all": "true",
    "autoReport.directory": <current_working_directory>,
    "autoReport.outputSerializedGraph": "false",
}
```

Which can also be done manually by exporting this variable in the benchmarking environment, if custom options are needed for this variable.

## Changelog <a name='changelog' ></a>

`June 2022`: This repository now implements *grouped* gathers and scatters: i.e. the compiler
is fully aware that each member of the batch can only communicate within its own
batch. To use this efficiently, we concatenate several graphs into each member
of the batch. The `packing_strategy_finder` plans the packing so as to minimise
padding. Using this grouping improves our throughput.

## Licensing <a name='licensing' ></a>

This example is licensed under the MIT license.

Copyright 2021 Graphcore

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
