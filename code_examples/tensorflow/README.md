# Code Examples for TensorFlow 1

This directory contains a number of code examples for how to use TensorFlow 1 with IPUs. These include full models as well as examples of how to measure performance, use multiple IPUs and implement custom ops in Poplar, among other things. This README provides an overview of all of the TensorFlow 1 examples.


## Complete examples of models

A number of complete examples of models implemented on the IPU are available.

- [UNet Industrial](unet_industrial) is an image segmentation model for industrial use cases.

- [Markov chain Monte Carlo](mcmc) methods are well known techniques for solving integration and optimisation problems in large dimensional spaces. Their applications include algorithmic trading and computational biology. This example uses the TensorFlow Probability library.

- [CosmoFlow](cosmoflow) is a deep learning model for calculating cosmological parameters. The model primarily consists of 3D convolutions, pooling operations, and dense layers.


## Other examples

- [Block sparsity](block_sparse): Examples for performing block-sparse computations on the IPU, including the Poplar code for two block-sparse custom ops (matrix multiplication and softmax), which are used to construct a simple MNIST example.

