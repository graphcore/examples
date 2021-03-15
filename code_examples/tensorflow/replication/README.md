# Graphcore

---
## TensorFlow Replication example

This example shows how to use replication in TensorFlow to train a very simple
MNIST Conv model.

Replication allows data-parallelism, where multiple copies of the same model
concurrently train on partitions of the same dataset, increasing throughput.
Each replica trains on a different (set of) IPU(s), building up a buffer of
calculated gradients. Periodically, the replicas synchronise and combine their
individual gradients in what's called an "all-reduce".

When using replication, ensure that:
1. The data pipeline has enough bandwidth to push the required volume of data;
2. Adjustments are made to the learning rate, as replication increases the
   effective batch size;
3. Adjustments are made to the calculation of throughput.

Note: Care should be used when finding the optimal number of replicas and size
of the max cross replica sum buffer - please see the code for a more detailed
explanation of how these parameters affect model memory and performance.

### File structure

* `replication.py` The main TensorFlow file showcasing replication.
* `README.md` This file.
* `test_replication.py` Script for testing this example.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the Getting Started guide for your IPU system.
   Make sure to run the enable.sh script and activate a Python virtualenv with the
   tensorflow-1 wheel from the Poplar SDK installed.

2) Run the script.

   `python replication.py`

#### Options

Run replication.py with the -h option to list all available command line options
