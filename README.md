# Graphcore code examples

This repository contains sample applications, code examples and tutorials for use with
Graphcore IPUs.

If you are interested in finding out more about Graphcore, including
getting preview access to IPUs to run these examples, please register
your interest [here](https://www.graphcore.ai/product_info).

Please note we are not currently accepting pull requests or issues on this
repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

The latest version of the documentation for the Poplar software stack, and other developer resources, is available at https://www.graphcore.ai/developer.

>  The code presented here requires using Poplar SDK 1.4.x

Please install and enable the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
Note that for SDK 1.3 and later you no longer need to source a separate `enable.sh` for the Graphcore drivers.

Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE referenced at the top level applies to the files in this repository.

## Repository contents

### Notable examples

| Example | Link |
| ------- | ---- |
| BERT | [code](applications/popart/bert) |
| BERT (TensorFlow) | [code](applications/tensorflow/bert) |
| DeepVoice3 (PopART) | [code](applications/popart/deep_voice) |
| CNN Training including ResNet, ResNeXt & EfficientNet (TensorFlow) | [code](applications/tensorflow/cnns/training) |
| CNN Inference including ResNet, MobileNet & EfficientNet (TensorFlow) | [code](applications/tensorflow/cnns/inference) |
| CNN Training & Inference including ResNet, ResNeXt & EfficientNet (PyTorch) | [code](applications/pytorch/cnns) |
| ResNet Training (PopART) | [code](applications/popart/resnet) |
| ResNext Inference (PopART) | [code](applications/popart/resnext_inference) |
| Recommendation using Autoencoders (TensorFlow) | [code](applications/tensorflow/autoencoder) |
| Sales forecasting example (TensorFlow) | [code](applications/tensorflow/sales_forecasting) |
| Contrastive Divergence VAE using MCMC methods (TensorFlow) | [code](applications/tensorflow/contrastive_divergence_vae) |
| Example reinforcement learning policy model (TensorFlow)| [code](applications/tensorflow/reinforcement_learning) |
| Click through rate: Deep Interest Network (TensorFlow) | [code](applications/tensorflow/click_through_rate) |
| Dynamic Sparsity: MNIST RigL (TensorFlow) | [code](applications/tensorflow/dynamic_sparsity/mnist_rigl) |
| Dynamic Sparsity: Autoregressive Language Modelling (TensorFlow) | [code](applications/tensorflow/dynamic_sparsity/language_modelling) |

### Application examples

The `applications/` folder contains example applications written in different frameworks targeting the IPU. See the READMEs in each folder for details on how to use these applications.

### Code examples

The `code_examples/` folder contains small code examples showing you how to use various software features when developing for IPUs. See the READMEs in each folder for details.

### Tutorials

The `tutorials/` folder contains tutorials to help you get started using the Poplar SDK and Graphcore tools. Currently, this contains:

* `tutorials/poplar` - A set of tutorials to introduce the Poplar graph programming framework and the PopLibs libraries.
* `tutorials/pytorch` - A tutorial to introduce the PyTorch framework support for the IPU.

The README files for the tutorials are best viewed on GitHub.

### Utilities

The `utils/` folder contains utilities libraries and scripts that are used across the other code examples. This includes:

* `utils/examples_tests` - Common Python helper functions for the repository's unit tests.
* `utils/benchmarks` - Common Python helper functions for running benchmarks on the IPU in different frameworks.
