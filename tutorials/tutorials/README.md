<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# Graphcore Tutorials

These tutorials provide hands-on programming exercises to enable you to
familiarise yourself with creating, running and profiling programs on the IPU.
They are part of the Developer resources provided by Graphcore:
<https://www.graphcore.ai/developer>.

Each of the tutorials contains its own README file with full instructions.

## Poplar

Poplar is the underlying C++ framework for developing and executing code on the Graphcore IPU.
It provides a generic interface on which the other frameworks are built.

- [Tutorial 1: Programs and Variables](poplar/tut1_variables)
- [Tutorial 2: Using PopLibs](poplar/tut2_operations)
- [Tutorial 3: Writing Vertex Code](poplar/tut3_vertices)
- [Tutorial 4: Profiling Output](poplar/tut4_profiling)
- [Tutorial 5: Matrix-vector Multiplication](poplar/tut5_matrix_vector)
- [Tutorial 6: Matrix-vector Multiplication Optimisation](poplar/tut6_matrix_vector_opt)

## TensorFlow 2 [![Gradient](../../gradient-badge.svg)](https://ipu.dev/3QA9C3e)

Getting started with the IPU:

- [Starter Tutorial: MNIST Training Example](../simple_applications/tensorflow2/mnist)
- [TensorFlow 2 Keras: How to Run on the IPU](tensorflow2/keras)

Exchanging data between the host and the IPU:

- [TensorFlow 2: How to Use Infeed/Outfeed Queues](tensorflow2/infeed_outfeed)

Debugging and analysis:

- [TensorFlow 2: How to Use TensorBoard](tensorflow2/tensorboard)

## PyTorch [![Gradient](../../gradient-badge.svg)](https://ipu.dev/3X896wa)

Getting started with the IPU:

- [Starter Tutorial: MNIST Training Example](../simple_applications/pytorch/mnist)
- [From 0 to 1: Introduction to PopTorch](pytorch/basics)

Exchanging data between the host and the IPU:

- [Efficient Data Loading with PopTorch](pytorch/efficient_data_loading)

Maximising compute on the IPU:

- [Half and Mixed Precision in PopTorch](pytorch/mixed_precision)

Using multiple IPUs and handling large models:

- [Parallel Execution Using Pipelining](pytorch/pipelining)
- [Fine-tuning BERT with HuggingFace and PopTorch](pytorch/finetuning_bert)

Debugging and analysis:

- [Observing Tensors in PopTorch](pytorch/observing_tensors)

Running a Hugging Face model on the IPU:

- [Fine-tuning a HuggingFace Vision Transformer (ViT) on the IPU Using a Local Dataset](pytorch/vit_model_training)

## PyTorch Geometric [![Gradient](../../gradient-badge.svg)](TODO)

Getting started:

- [Tutorial 1: At a glance](pytorch_geometric/1_at_a_glance)
- [Tutorial 2: A worked example](pytorch_geometric/2_a_worked_example)

A closer look at batching small graphs ready for the IPU:

- [Tutorial 3: Small graph batching with padding](pytorch_geometric/3_small_graph_batching_with_padding)
- [Tutorial 4: Small graph batching with packing](pytorch_geometric/3_small_graph_batching_with_packing)

## PopVision

PopVision is Graphcore's suite of graphical application analysis tools.

- [Instrumenting applications and using the PopVision System Analyser](popvision/system_analyser_instrumentation)
- [Accessing profiling information with libpva](popvision/libpva)
- [Reading application instrumentation from PVTI files](popvision/reading_pvti_files)
- Profiling output with the PopVision Graph Analyser is currently included in [Poplar Tutorial 4: profiling output](poplar/tut4_profiling)

## PopXL and popxl.addons

PopXL and popxl.addons are Graphcore frameworks which provide low level
control of the IPU through an expressive
Python interface designed for machine learning applications.

- [Tutorial 1: Basic Concepts](popxl/1_basic_concepts)
- [Tutorial 2: Custom Optimiser](popxl/2_custom_optimiser)

Improving performance and optimising throughput:

- [Tutorial 3: Data Parallelism](popxl/3_data_parallelism)
- [Tutorial 4: Pipelining](popxl/4_pipelining)
- [Tutorial 5: Remote Variables](popxl/5_remote_variables_and_rts)
- [Tutorial 6: Phased Execution](popxl/6_phased_execution)

## Standard tools

In this folder you will find explanations of how to use standard deep learning tools
with the Graphcore IPU. Guides included are:

- [Using IPUs from Jupyter Notebooks](standard_tools/using_jupyter)
- [Using VS Code with the Poplar SDK and IPUs](standard_tools/using_vscode)
