<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# PyTorch Feature Examples

This directory contains code examples showing how to use PyTorch with Graphcore's IPUs. These include full models as well as examples of how to use pre-trained models.

## Complete examples of models

- [Octave Convolutions](octconv) are a novel convolutional layer in neural networks. This example shows an implementation of how to train the model and run it for inference.

## Using multiple IPUs

- [Distributed Training using PopDist](popdist): This shows how to make an application ready for distributed training by using the PopDist API, and how to launch it with the PopRun distributed launcher.

## Custom operators

- [Using Custom Operators](custom_op) shows the use of a custom operator in PopTorch. This example shows an implementation of the Leaky ReLU custom operator in the training of a simple model.
