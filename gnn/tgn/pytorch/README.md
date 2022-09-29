# Temporal Graph Networks

This directory contains a PyTorch implementation of [Temporal Graph Networks](https://arxiv.org/abs/2006.10637) to train on IPU.
This implementation is based on [`examples/tgn.py`](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/tgn.py) from PyTorch-Geometric.

## Running on IPU

### Setting up the environment
Install the Poplar SDK following the [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for the IPU system. 
Source the `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv with PopTorch installed.

Now install the dependencies of the TGN model:
```bash
pip install -r requirements.txt
```

### Train the model
To train the model run 
```bash
python train.py
```

The following flags can be used to adjust the behaviour of `train.py`

--data: directory to load/save the data (default: data/JODIE) <br>
-t, --target: device to run on (choices: {ipu, cpu}, default: ipu) <br>
-d, --dtype: floating point format (default: float32) <br>
-e, --epochs: number of epochs to train for (default: 50) <br>
--lr: learning rate (default: 0.0001) <br>
--dropout: dropout rate in the attention module (default: 0.1) <br>
--optimizer, Optimizer (choices: {SGD, Adam}, default: Adam) <br>

### License
This application is licensed under the MIT license, see the LICENSE file at the top-level of this repository.

This directory includes derived work from the PyTorch Geometric repository, https://github.com/pyg-team/pytorch_geometric by Matthias Fey and Jiaxuan You, published under the MIT licenes