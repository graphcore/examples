# Graphcore

---
## PopART MNIST Training Demo

This example trains a network on the MNIST dataset using PopART.

### File structure

* `popart_mnist.py` The main PopART program that uses a linear network.
* `get_data.sh` Script to fetch the images and labels.
* `README.md` This file.
* `requirements.txt` Specifies the required modules.

### How to use this demo

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU System.
   Make sure to source the `enable.sh` scripts for poplar and popart.

   The PopART Python API only supports Python 3. It is recommended to use a virtualenv.

   Install the required modules:

       pip3 install -r requirements.txt

2) Download the data.

       ./get_data.sh

   This will create and populate a `data` directory.

4) Run the graph.

       python3 popart_mnist.py

#### Options
The `popart_mnist.py` has a few command line options.

`-h`                  Show usage information.

`--batch-size`        Sets the batch size. This must be a multiple of the
replication factor.

`--batches-per-step`  Number on mini-batches to perform on the device before returning to the host.

`--epochs`            Number of epoch to train for.

`--num-ipus`          Number of IPUs to use.

`--pipeline`          Pipeline the model over IPUs. Only valid for this model
if the number of IPUs is double the replication factor.

`--replication-factor` Number of times to replicate the graph to perform data parallel training. This must be a factor of the number of IPUs. Defaults to 1.

`--simulation`        Run with the IPU_MODEL device instead of hardware.

`--log-graph-trace`   Turn on IR logging to display the graph's ops.


Examples:

     python popart_mnist.py --num-ipus 2 --pipeline

     python popart_mnist.py --num-ipus 4 --replication-factor 2 --pipeline
