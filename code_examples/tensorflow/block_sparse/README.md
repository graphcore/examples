# Block-Sparse library

> Copyright (c) 2020 Graphcore Ltd. All rights reserved.

## Dependencies

Poplar SDK version 1.4 or later

spdlog library. Tested on version 1:0.16.3-1 for ubuntu

## Quick start guide

### Prepare the environment

##### 1) Download the Poplar SDK

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` script for Poplar.

##### 2) Python

Create a virtualenv and install the required packages:

```bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

Add a path to block_sparse directory:
```bash
export PYTHONPATH=.../block_sparse:$PYTHONPATH
```

### Create libblock_sparse.so with provided makefile
```bash
make all
```

### Test basic examples
```bash
# To run block-sparse matmul test app:
python3 bsmatmul_test.py --lhs-rows=64 --lhs-cols=64 --rhs-cols=64 --sparsity=0.5

# To run block-sparse softmax test app:
python3 bssoftmax_test.py --rows=64 --cols=64 --sparsity=0.5

# To run all unit tests:
python3 -m pytest
```

### Run MNIST example
```bash
cd examples/mnist
python3 mnist_tf.py
```
