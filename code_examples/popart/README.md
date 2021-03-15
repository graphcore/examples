# PopART Code Examples

This directory contains a number of code examples demonstrating how to use the Poplar Advanced Runtime (PopART). These include full models as well as examples of how to use multiple IPUs, implement custom ops in Poplar and other key features provided by PopART.


## Contents

### Simple Models and Benchmarks

- [Simple MNIST Examples](mnist): Contains 2 simple models, 1 linear and 1 using convolutions trained on the MNIST dataset.

- [Kernel Synthetic Benchmarks](kernel_benchmarks): Contains synthetic benchmarks for models with two types of layer (LSTM and 2D Convolution) and synthetic data in training and inference.


### Multi IPU Examples

- [Sharding a Model over Multiple IPUs](sharding): This demo shows how to "shard" (split) a model over multiple IPUs using PopART.

- [Pipelining a Model over Multiple IPUs](pipelining): This demo shows how to use pipelining in PopART on a very simple model consisting of two dense layers.

- [Utilising Streaming Memory with Phased Execution](phased_execution): This example runs a network in inference mode over two IPUs by splitting it in several execution phases and keeping the weights in Streaming Memory.


### Further Examples

- [Custom Operators](custom_operators): This directory contains two example implementations of custom operators for PopART (Cube and LeakyReLU). Both examples create an operation definition with forward and backward parts, and include a simple inference script to demonstrate using the operators.

- [Block Sparsity](block_sparse): Examples of performing block-sparse computations on the IPU, including the Poplar code for two block-sparse custom ops (matrix multiplication and softmax), which are used to construct a number of examples and test scripts, including a simple MNIST example.

- [Data Callbacks](callbacks): This example creates a simple computation graph and uses callbacks to feed data and retrieve the results. Time between host-device transfer and receipt of the result on the host is computed and displayed for a range of different data sizes.

- [Distributed MNIST with Horovod Training Demo](distributed_training): This example uses distributed training with a Horovod PopART extension to train a network on the MNIST dataset.

- [Automatic and Manual Recomputing](recomputing): This example shows how to use manual and automatic recomputation in popART with a seven layer DNN and generated data.
