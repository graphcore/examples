<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# TensorFlow 2 Feature Examples

This directory contains several examples showing how to use TensorFlow 2 on the IPU.

- [IPUEstimator](ipu_estimator): This example shows how to train a model to sort images from the CIFAR-10 dataset using the IPU implementation of the TensorFlow Estimator API.

- [Embeddings](embeddings): These examples train an IPU model with an embedding layer and an LSTM to predict the sentiment of an IMDB review.

- [Inspecting Tensors Using Custom Outfeed Layers and a Custom Optimizer](inspecting_tensors): This example trains a choice of simple fully connected models on the MNIST numeral data set and shows how tensors (containing activations and gradients) can be returned to the host via outfeeds for inspection.

- [Distributed Training and Inference](popdist): This shows how to prepare an application for distributed training and inference by using the PopDist API, and how to launch it with the PopRun distributed launcher.

- [Recomputation Checkpoints](recomputation_checkpoints): This example demonstrates using checkpointing of intermediate values to reduce live memory peaks with a simple Keras LSTM model.
