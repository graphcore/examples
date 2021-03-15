# Code Examples for TensorFlow 1

This directory contains a number of code examples for how to use TensorFlow 1 with IPUs. These include full models as well as examples of how to measure performance, use multiple IPUs and implement custom ops in Poplar, among other things. This README provides an overview of all of the TensorFlow 1 examples.


## Complete examples of models

A number of complete examples of models implemented on the IPU are available.
 
- [Classifying hand-written digits](mnist) from the MNIST dataset is a well-known example of a basic machine learning task. An example of its implementation on IPUs can be found in `mnist`. This example also shows how to use `ipu.dataset_benchmark` to determine the maximum achievable throughput for a given dataset.

- [UNet Industrial](unet_industrial) is an image segmentation model for industrial use cases.

- [Markov chain Monte Carlo](mcmc) methods are well known techniques for solving integration and optimisation problems in large dimensional spaces. Their applications include algorithmic trading and computational biology. This example uses the TensorFlow Probability library.

- [CosmoFlow](cosmoflow) is a deep learning model for calculating cosmological parameters. The model primarily consists of 3D convolutions, pooling operations, and dense layers.


## Performance and profiling

- [Kernel benchmarking](kernel_benchmarks): Code for benchmarking the performance of some selected neural network layers.

- [I/O benchmarking](mnist): The MNIST example in `mnist` shows how to use `ipu.dataset_benchmark` to determine the maximum achievable throughput for a given dataset.

- [Profiling](report_generation): Code demonstrating how to generate text-based reports on the performance of your model.


## Using multiple IPUs

Simple examples demonstrating and explaining different ways of using multiple IPUs are provided. [Pipelining](pipelining) and [replication](replication) are generally used to parallelise and speed up training, whereas [sharding](sharding) is generally used to simply fit a model in memory.


## Custom ops

- [Custom op example](custom_op): Code that demonstrates how to define your own custom op using Poplar and PopLibs and use it in TensorFlow 1.

- [Custom op example with gradient](custom_gradient): Code that demonstrates how to define your own custom op using Poplar and PopLibs and use it in TensorFlow 1. Also shows how to define the gradient of your custom op so that you can use automatic differentiation and operations that depend on it, such as the `minimize` method of an optimizer.


## Other examples

- [IPUEstimator](ipuestimator): Example of using the IPU implementation of the TensorFlow Estimator API.

- [Block sparsity](block_sparse): Examples for performing block-sparse computations on the IPU, including the Poplar code for two block-sparse custom ops (matrix multiplication and softmax), which are used to construct a simple MNIST example.

- [Configuring IPU connections](connection_type): A code example which demonstrates how to use `ipu.utils.set_ipu_connection_type` to control if and when the IPU device is acquired.
