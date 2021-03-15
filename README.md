# Graphcore code examples

This repository contains sample applications, code examples and tutorials for use with
Graphcore IPUs.

If you are interested in finding out more about Graphcore, including
getting preview access to IPUs to run these examples, please register
your interest [here](https://www.graphcore.ai/product_info).

Please note we are not currently accepting pull requests or issues on this
repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

The latest version of the documentation for the Poplar software stack, and other developer resources, is available at https://www.graphcore.ai/developer.

>  The code presented here requires using Poplar SDK 2.0.x

Please install and enable the Poplar SDK following the instructions in the Getting Started guide for your IPU system.

Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE referenced at the top level applies to the files in this repository.

## Repository contents

### Notable examples

| Example | Link |
| ------- | ---- |
| BERT (PopART) | [code](applications/popart/bert) |
| BERT (TensorFlow) | [code](applications/tensorflow/bert) |
| BERT (Pytorch) | [code](applications/pytorch/bert) |
| DeepVoice3 (PopART) | [code](applications/popart/deep_voice) |
| CNN Training including ResNet, ResNeXt & EfficientNet (TensorFlow) | [code](applications/tensorflow/cnns/training) |
| CNN Inference including ResNet, MobileNet & EfficientNet (TensorFlow) | [code](applications/tensorflow/cnns/inference) |
| CNN Training & Inference including ResNet, ResNeXt & EfficientNet (PyTorch) | [code](applications/pytorch/cnns) |
| ResNext Inference (PopART) | [code](applications/popart/resnext_inference) |
| Recommendation using Autoencoders (TensorFlow) | [code](applications/tensorflow/autoencoder) |
| Sales forecasting example (TensorFlow) | [code](applications/tensorflow/sales_forecasting) |
| Contrastive Divergence VAE using MCMC methods (TensorFlow) | [code](applications/tensorflow/contrastive_divergence_vae) |
| Example reinforcement learning policy model (TensorFlow)| [code](applications/tensorflow/reinforcement_learning) |
| Click through rate: Deep Interest Network (TensorFlow) | [code](applications/tensorflow/click_through_rate) |
| Click through rate: Deep Interest Evolution Network (TensorFlow) | [code](applications/tensorflow/click_through_rate) |
| Dynamic Sparsity: MNIST RigL (TensorFlow) | [code](applications/tensorflow/dynamic_sparsity/mnist_rigl) |
| Dynamic Sparsity: Autoregressive Language Modelling (TensorFlow) | [code](applications/tensorflow/dynamic_sparsity/language_modelling) |

### Application examples

The [applications/](applications) folder contains example applications written in different frameworks targeting the IPU. See the READMEs in each folder for details on how to use these applications.

### Code examples

The [code_examples/](code_examples) folder contains small code examples showing you how to use various software features when developing for IPUs. See the READMEs in each folder for details.

### Tutorials

The [tutorials/](tutorials) folder contains tutorials to help you get started using the Poplar SDK and Graphcore tools. Currently, this contains:

* [tutorials/poplar](tutorials/poplar) - A set of tutorials to introduce the Poplar graph programming framework and the PopLibs libraries.
* [tutorials/pytorch](tutorials/pytorch) - A tutorial to introduce the PyTorch framework support for the IPU.
* [tutorials/tensorflow1](tutorials/tensorflow1) - A set of tutorials to introduce the TensorFlow 1 framework support for the IPU.
* [tutorials/tensorflow2](tutorials/tensorflow2) - A set of tutorials to introduce the TensorFlow 2 framework support for the IPU.
* [tutorials/popvision](tutorials/popvision) - A set of tutorials to introduce PopVision, our suite of graphical application analysis tools.

A complete list of available tutorials can be found in the [tutorials/](tutorials) folder.

The README files for the tutorials are best viewed on GitHub.

### Utilities

The [utils/](utils) folder contains utilities libraries and scripts that are used across the other code examples. This includes:

* [utils/examples_tests](utils/examples_tests) - Common Python helper functions for the repository's unit tests.
* [utils/benchmarks](utils/benchmarks) - Common Python helper functions for running benchmarks on the IPU in different frameworks.
