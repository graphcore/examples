# Graphcore

---
## PyTorch(PopTorch) MNIST Training Demo

This example demonstrates how to train a network on the MNIST dataset using PopTorch.

### File structure

* `mnist_poptorch.py` The main file.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

    Install the `poplar-sdk` following the README provided. Make sure to run the `enable.sh` scripts and activate a Python virtualenv with poptorch installed.

    Then install the package requirements:

       pip install -r requirements.txt


2) Run the graph. Note that the PopTorch Python API only supports Python 3.
Data will be automatically downloaded using torch vision utils.

       python3 mnist_poptorch.py

#### Options
The program has a few command-line options:

`-h` Show usage information.

`--batch-size`        Sets the batch size for training.

`--batches-per-step`  Number on mini-batches to perform on the device before returning to the host.

`--test-batch-size`   Sets the batch size for inference.

`--epochs`            Number of epoch to train for.

`--lr`                Learning rate of the optimizer.

