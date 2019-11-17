# Graphcore

---
## PopART MNIST Training Demo

This example trains a network on the MNIST dataset using a port
of ONNX that targets Graphcore's Poplar libraries.

### File structure

* `popart_mnist.py` The main PopART file using a linear network.
* `popart_mnist_conv.py` The main PopART file using a convolutional network.
* `get_data.sh` Script to fetch the images and labels.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
    scripts for poplar, gc_drivers (if running on hardware) and popart.

2) Download the data.

       ./get_data.sh

  This will create and populate a `data/` directory.

4) Run the graph. Note that the PopART Python API only supports Python 3.

       python3 popart_mnist.py

or

       python3 popart_mnist_conv.py



#### Options
The program has a few command-line options:

`-h` Show usage information.

`--batch-size`        Sets the batch size.

`--batches-per-step`  Number on mini-batches to perform on the device before returning to the host.

`--epochs`            Number of epoch to train for.

`--simulation`        Run with the IPU_MODEL device instead of hardware.

`--log-graph-trace`   Turn on IR logging to display the graph's ops.


