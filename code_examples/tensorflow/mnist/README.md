# Graphcore

---
## Basic Model Demo for MNIST dataset

This demo shows how to use one IPU for a simple Tensorflow model with MNIST dataset.
The `IPUInfeedQueue` is used to stream input data set for the IPU.
The `ipu.dataset_benchmark` tool allows to obtain the maximum achievable throughput of the infeed.

### File structure

* `mnist_tf.py` The main python script.
* `README.md` This file.

### How to use this demo
1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the Getting Started guide for your IPU system.
   Make sure to run the `enable.sh` script and activate a Python virtualenv with the tensorflow-1 wheel from the Poplar SDK installed.

2) Install the package requirements

   `pip install -r requirements.txt`

3) Train and test the model:

   `python3 mnist_tf.py`
