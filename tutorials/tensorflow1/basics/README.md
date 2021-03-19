# Introduction to TensorFlow 1 on the IPU

This series of tutorials covers the basics of porting a TensorFlow 1 model to run efficiently on Graphcore IPUs using the Python API. It uses TensorFlow 1.15, the version of TensorFlow 1 included with the Poplar SDK. It assumes familiarity with basic machine learning concepts and TensorFlow 1, including the `tf.data` API used for building data pipelines. There are currently two tutorials:

1. Porting a simple example
2. Loops and data pipelines

In each of these tutorials, we walk through a full code example, creating and training a neural network to identify items of clothing from the Fashion-MNIST dataset. This dataset is downloaded using the TensorFlow API. See the [TensorFlow API documentation](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/datasets/fashion_mnist/load_data) for the license details for this dataset.

Each tutorial lives in its own directory `tutorial_X`, containing a file `README.md` with the walkthrough itself, the code example `example_X.py`, and anything else necessary for the tutorial. Each snippet in each tutorial is numbered to help you find it in the code.

## Before you start

These tutorials are reasonably self-contained and can be followed without access to Graphcore technology. However, to make full use of them, you will need access to IPU hardware and the latest Poplar SDK. For software installation and setup details, please see the Getting Started guide for your hardware setup, available [here](https://docs.graphcore.ai/en/latest/hardware.html#getting-started).

## Other useful resources

Here are some other useful resources:

[TensorFlow Docs](https://docs.graphcore.ai/en/latest/software.html#tensorflow)
- All Graphcore documentation specifically relating to TensorFlow.

[IPU TensorFlow Code Examples](https://github.com/graphcore/examples/tree/master/code_examples/tensorflow)
- Lots of examples of different use cases of TensorFlow on the IPU.

If you need any more personal help, please do not hesitate to contact our [support desk](https://www.graphcore.ai/support).