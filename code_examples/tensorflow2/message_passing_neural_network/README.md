## Changelog

N/A (initial commit)

## Overview of the Architectures

In this repository, we use the Graphcore IPU to implement a few popular graph neural network architectures.

For training, `--model==graph_isomorphism` uses a Graph Isomorphism Network (GIN)[1]. `--model==graph_network` uses a Graph Network[2], and `--model=interaction_network` implements an Interaction Network[3]. These are all used to predict chemical properties of molecules on the IPU.

The script `run_training.py` runs and evaluates training. The script `benchmark.py` uses synthetic data for evaluating the throughput of the model. This repository is written in TensorFlow 2 and uses keras extensively.

This repository supports training, evaluation, and benchmarking the throughput.

<!--
-->
*[1] Xu, Keyulu, et al. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).*

*[2] Battaglia, Peter W., et al. "Relational inductive biases, deep learning, and graph networks." arXiv preprint arXiv:1806.01261 (2018).*

*[3] Gilmer, Justin, et al. "Neural message passing for quantum chemistry." International conference on machine learning. PMLR, 2017.*


## Setup

Install dependencies with: 

`pip install --user -r requirements.txt`

#### Quick-start

To quickly check that the IPU is set up and working, try: 
`python benchmark.py`, which will run on synthetic data.

To train a Graph Isomorphism Network, run:

`python run_training.py  --dtype=float32 --model=graph_isomorphism --n_graph_layers=5  --nodes_dropout=0.1 `

This should get a test AUC of slightly above 0.75.

## Dataset

This example trains using the Open Graph Benchmark [4] — specifically, the `molhiv` dataset. This contains over 40,000 molecules. The machine learning task is to predict whether a molecule inhibits HIV replication.

The leaderboard for this task is found [here](https://ogb.stanford.edu/docs/leader_graphprop/).

*[4] Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." arXiv preprint arXiv:2005.00687 (2020).*

### Model Details

To integrate the use of edges (i.e. the information from atomic bonds), we follow [5] and embed the edges afresh at each layer. These embedded edges are combined with the neighborhood aggregation from the nodes. 

We used similar parameters to the example in [5], but use Layer Normalization instead of Batch Normalization at the hidden layers of the multi-layer perceptrons.

*[5] Hu, Weihua, et al. "Strategies for pre-training graph neural networks." arXiv preprint arXiv:1905.12265 (2019).*

### Performance

We compare our model with the GIN implemented on the [OGB leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv). As per the instructions, training is performed 10 times and the results reported.

The test results are consistent with the reported results.

| Model | Parameters | Test ROC-AUC | Validation ROC-AUC |
|----|----|----|----|
| GIN on leaderboard | 1,885,206| 0.7558 ± 0.0140 | 0.8232 ± 0.0090 | 
| IPU GIN model | 1,708,106 | 0.7588 ± 0.0131 | 0.7799 ± 0.014| 

## Licensing

This example is licensed under the MIT license.

Copyright 2021 Graphcore

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


