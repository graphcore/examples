# Code Examples for PyTorch

This directory contains a number of code examples showing how to use PyTorch with Graphcore's IPUs. These include full models as well as examples of how to use pre-trained models.

## Complete examples of models

A number of complete examples of models implemented on the IPU are available.

- [Classifying hand-written digits](mnist) from the MNIST dataset is a well-known example of a basic machine learning task. 

- [Octave Convolutions](octconv) are a novel convolutional layer in neural networks. This example shows an implementation of how to train the model and run it for inference.

## Pre-trained models

- [Hugging Face's BERT](bert) is a pre-trained BERT model made available by Hugging Face and which is implemented in PyTorch. This example consists of running one of the pre-trained BERT model on an IPU for an inference session.

## Other examples

- [PopART's MNIST](popart_api/mnist) is an example on how to export a PyTorch model as an ONNX file and reuse this file with Graphcore's PopART.

