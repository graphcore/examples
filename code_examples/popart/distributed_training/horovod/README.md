# Graphcore

---
## Distributed PopART MNIST with Horovod Training Demo

This example trains a network on the MNIST dataset using a port
of ONNX that targets Graphcore's Poplar libraries.

### File structure

* `horovod_popart_mnist.py` The main PopART file using a linear network using distributed training.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Create a Python 3.6 environment with virtualenv: `virtualenv python36 -p python3.6`
   Download and unpack the Poplar SDK found on [Downloads](https://downloads.graphcore.ai/).
   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` scripts for Poplar and PopART.
   Install the Horovod PopART extension from the Poplar SDK directory by running: `pip3 install horovod-*-cp36m-linux_x86_64.whl`

2) Download the data.

       ./get_data.sh

  This will create and populate a `data/` directory.

4) Run distributed training with Horovod.

       horovodrun -np 3 -H localhost:3 python3 horovod_popart_mnist.py

  This runs three independent processes that will perform AllReduce and Broadcast operations to synchronize the distributed training.

  Alternatively we can use the [Gloo](https://github.com/facebookincubator/gloo) backend for the collective operations:

       horovodrun --gloo -np 3 -H localhost:3 python3 horovod_popart_mnist.py

#### Options
The program has a few command-line options:

`-h` Show usage information.

`--batch-size`        Sets the batch size.

`--batches-per-step`  Number on mini-batches to perform on the device before returning to the host.

`--epochs`            Number of epoch to train for.

`--simulation`        Run with the IPU_MODEL device instead of hardware.

`--log-graph-trace`   Turn on IR logging to display the graph's ops.

`--syn-data-type`     Use synthetic data for the training without performing host IO. Possible values are "zeros" and "random-normal"


#### Unit tests
The unit tests can be run with the following command from the script folder: `python3 -m pytest test_horovod_popart_mnist.py`
