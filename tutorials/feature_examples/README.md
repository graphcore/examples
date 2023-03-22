<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# Graphcore Feature Examples

The code examples demonstrate features which will enable you to make the most of
the IPU. They are part of the Developer resources provided by Graphcore:
<https://www.graphcore.ai/developer>.

Each of the examples contains its own README file with full instructions.

## Poplar

Exchange data between host and IPU efficiently:

- [Prefetching Callbacks](poplar/prefetch): A demonstration of prefetching data when a
  program runs several times.

Demonstrate advanced features of Poplar:

- [Advanced Features](poplar/advanced_example): An example demonstrating several
  advanced features of Poplar, including saving and restoring Poplar
  executables, moving I/O into separate Poplar programs, and using our PopLibs
  framework.

## TensorFlow 2

Debugging and analysis:

- [Inspecting Tensors Using Custom Outfeed Layers and a Custom Optimizer](tensorflow2/inspecting_tensors): An example that shows
  how outfeed queues can be used to return activation and gradient tensors to
  the host for inspection.

Use estimators:

- [IPUEstimator](tensorflow2/ipu_estimator): An example showing how to use the
  IPUEstimator to train and evaluate a simple CNN.

Specific layers:

- [Embeddings](tensorflow2/embeddings): An example of a model with an embedding
  layer and an LSTM, trained on the IPU to predict the sentiment of an IMDB
  review.

- [Recomputation Checkpoints](tensorflow2/recomputation_checkpoints): An example
  demonstrating the checkpointing of intermediate values to reduce live memory
  peaks with a simple Keras LSTM model.

Efficiently use multiple IPUs and handle large models:

- [Distributed Training and Inference](tensorflow2/popdist): This shows how to prepare a TensorFlow 2 application for distributed training and inference by using the PopDist API, and how to launch it with the PopRun distributed launcher.

## PyTorch

Efficiently use multiple IPUs and handle large models:

- [Distributed Training](pytorch/popdist): An example showing how to prepare a PyTorch application
  for distributed training and inference using the PopDist library, and
  how to launch it with the PopRun distributed launcher.

Define custom operators:

- [Using Custom Operators](pytorch/custom_op): An example showing how to create a PopART
  custom operator available to PopTorch and how to use it in a model.

Specific layers:

- [Octave Convolutions](pytorch/octconv): An example showing how to use Octave Convolutions
  in PopTorch training and inference models.
