# Message Passing Neural Network (MPNN)
MPNN implementations (Graph Isomorphism Network, Graph Network and Interaction Network), optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| TensorFlow 2 | GNNs | MPNN | MolHIV |  | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required| <p style="text-align: center;">❌ | [How powerful are graph neural networks?](https://arxiv.org/abs/1810.00826), [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261), [Neural message passing for quantum chemistry](https://arxiv.org/abs/1704.01212) |


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

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the TensorFlow 2 and IPU TensorFlow add-ons wheels:
```bash
cd <poplar sdk root dir>
pip3 install tensorflow-2.X.X...<OS_arch>...x86_64.whl
pip3 install ipu_tensorflow_addons-2.X.X...any.whl
```
For the CPU architecture you are running on

4. Build the custom ops:
```bash
cd static_ops && make
```


More detailed instructions on setting up your TensorFlow 2 environment are available in the [TensorFlow 2 quick start guide](https://docs.graphcore.ai/projects/tensorflow2-quick-start).

## Dataset setup
### MolHIV
Download the "molhiv" dataset via [the ogb repository](https://github.com/snap-stanford/ogb). The leaderboard for this task is found [here](https://ogb.stanford.edu/docs/leader_graphprop/).

*[1] Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." arXiv preprint arXiv:2005.00687 (2020).*


## Custom training

### Model Details

To integrate the use of edges (i.e. the information from atomic bonds), we follow [5] and embed the edges afresh at each layer. These embedded edges are combined with the neighborhood aggregation from the nodes.

We used similar parameters to the example in [5], but use Layer Normalization instead of Batch Normalization at the hidden layers of the multi-layer perceptrons.

*[2] Hu, Weihua, et al. "Strategies for pre-training graph neural networks." arXiv preprint arXiv:1905.12265 (2019).*

### Performance

We compare our model with the GIN implemented on the [OGB leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv). As per the instructions, training is performed 10 times and the results reported.

The command used is: `python run_training.py --micro_batch_size=24 --cosine_lr=True --epochs=150  --nodes_dropout=.1`, which uses FP16 training.

The test results are consistent with the reported results.

| Model | Parameters | Test ROC-AUC | Validation ROC-AUC |
|----|----|----|----|
| GIN on leaderboard | 1,885,206| 0.7558 ± 0.0140 | 0.8232 ± 0.0090 |
| IPU GIN model | 1,708,106 | 0.7574 ± 0.0215 | 0.7915 ± 0.0114|

## Other features

### Grouped gathers/scatters
This repository now implements *grouped* gathers and scatters. The compiler
is fully aware that each member of the batch can only communicate within its own
batch. To use this efficiently, we concatenate several graphs into each member
of the batch. The `packing_strategy_finder` plans the packing so as to minimise
padding. Using this grouping improves our throughput.


## Licensing

This example is licensed under the MIT license.

Copyright 2022 Graphcore

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
