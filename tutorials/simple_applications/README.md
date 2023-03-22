<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->

# Graphcore Simple Applications

This directory contains a number of basic applications written in different frameworks targeting the IPU.

## Poplar

- [Simple MNIST Training Example](poplar/mnist): this example shows how to build a training model to classify digits from the MNIST dataset.

## TensorFlow 2

- [Simple MNIST Training Example](tensorflow2/mnist): This example trains a simple 2-layer fully connected model on the MNIST numeral data set.

## PyTorch

### Complete examples of models

- [Classifying Hand-written Digits](pytorch/mnist) from the MNIST dataset is a well-known example of a basic machine learning task.

### Pre-trained models

- [Hugging Face's BERT](pytorch/bert) is a pre-trained BERT model made available by Hugging Face and which is implemented in PyTorch. This example consists of running one of the pre-trained BERT model on an IPU for an inference session.
