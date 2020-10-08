# Graphcore

---
## PyTorch PopART MNIST Training Demo

This example demonstrates how to train a network defined in PyTorch on the MNIST dataset using a popart backend which is a port of ONNX that targets Graphcore's Poplar Libraries.

### File structure

* `pytorch_popart_mnist.py` The main file.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
    scripts for poplar and popart.


2) Run the graph. Note that the PopART Python API only supports Python 3.
Data will be automatically downloaded using torch vision utils.

       python3 pytorch_popart_mnist.py

#### Options
The program has a few command-line options:

`-h` Show usage information.

`--batch-size`        Sets the batch size.

`--batches-per-step`  Number on mini-batches to perform on the device before returning to the host.

`--epochs`            Number of epoch to train for.

`--simulation`        Run with the IPU_MODEL device instead of hardware.

`--log-graph-trace`   Turn on IR logging to display the graph's ops.


