# Graphcore: Sharding Pipeline Stages using Concurrent Pipelines

---
## Concurrent Pipeline Support in TensorFlow
TensorFlow on IPU supports [concurrent pipelines](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/tensorflow/perf_training.html#concurrent-pipeline-stages).
These are single pipeline stages that use more than a single IPU to allow tensor parallel (sharded) computations to be defined in that stage. This code example shows
how to use this feature to implement a tensor parallel tied embedding where the embedding lookup, projection, and final softmax operations are sharded across multiple
IPUs. It also contains a sharded ops library in `sharded.py` that can be used to build other applications and an MNIST example showing how to use the library in such
an application.

### File structure

* `custom_ops` A TensorFlow Python module for accessing IPU specific sharded ops.
* `test` Test scripts and tools.
* `run_sharded_mnist.py` Simple example of pipelined MNIST that uses a sharded matmul in its final dense layer.
* `README.md` This file.

### How to use this example

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the enable.sh script for poplar and install the Graphcore TensorFlow 1.15.x wheel.

2) Install required pip modules:

```bash
pip install -r requirements.txt
```

4) Build the custom ops and then run the Python code. The command below runs a simple test of a tied embedding pipeline which checks
the loss and embedding matrix gradient matches a JAX based CPU implementation.

Build the sharded custom ops module and run one of the tests:
```bash
make -j10
python3 tests/sharded_embedding_tool.py --ipus 4 --vocab-size 8000 --feature-size 768 --sequence-length 256
```

If you have previously built this module using a different SDK version you must run ```make clean``` before re-running ```make```.

## Concurrent Pipelined MNIST example

As well as the tests there is also an example of using the feature to train MNIST in `run_sharded_mnist.py`. This can be used to
analyse how the library behaves in a training pipeline (the unit tests only check loss and gradients). The program allows you to
compare a standard pipeline with the concurrent one. If you run the following commands you should see that they both train similarly
(same loss profile):

```bash
python3 run_sharded_mnist.py --pipeline-mode basic
python3 run_sharded_mnist.py --pipeline-mode concurrent
```

The following is a schematic representation of the basic MNIST pipeline model
in this example which splits layers across two IPUs:
-------------------------- Basic Pipeline -----------------------------
IPU0: inputs -> MLP \
IPU1:                |-> Classifier -- SoftmaxCE -- Loss

However, in the concurrent pipeline case the final matrix multiply (classifier layer) and the following softmax are
executed tensor parallel in concurrent stages:
---------------------------- Concurrent Pipeline --------------------------------
IPU0: inputs -> MLP \ -> Classifier (top rows) -- SoftmaxCE(top)        |-> Combined Loss
IPU1:                |-> Classifier (bottom rows) -- SoftmaxCE(bottom) /
