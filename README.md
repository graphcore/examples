# Graphcore code examples

This repository contains sample applications and code examples for use with
Graphcore IPUs.

If you are interested in finding out more about Graphcore, including
getting preview access to IPUs to run these examples, please register
your interest [here](https://www.graphcore.ai/product_info).

Please note we are not currently accepting pull requests or issues on this
repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

The latest version of the documentation for the Poplar software stack, and other developer resources, is available at https://www.graphcore.ai/developer.

>  The code presented here requires using POPLAR SDK v1.2.x

Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE referenced at the top level applies to the files in this repository.

## Repository contents

### Notable examples

| Example | Link |
| ------- | ---- |
| BERT | [code](applications/popart/bert) |
| DeepVoice3 | [code](applications/popart/deep_voice) |
| CNN Training (including ResNet, ResNeXt & EfficientNet | [code](applications/tensorflow/cnns/training) |
| CNN Inference (including ResNet, MobileNet & EfficientNet | [code](applications/tensorflow/cnns/inference) |
| ResNext Inference | [code](applications/popart/resnext_inference) |
| Recommendation using Autoencoders | [code](applications/tensorflow/autoencoder) |
| Sales forecasting example | [code](applications/tensorflow/sales_forecasting) |
| Recurrent layer kernel benchmarks | [code](code_examples/tensorflow/kernel_benchmarks) |
| Constrastive Divergence VAE using MCMC methods | [code](applications/tensorflow/contrastive_divergence_vae) |
| Example reinforcement learning policy model | [code](applications/tensorflow/reinforcement_learning) |
| MCMC methods example | [code](code_examples/tensorflow/mcmc) |
| CosmoFlow example using 3D Convolutions | [code](code_examples/tensorflow/cosmoflow) |
| Grouped convolution kernel benchmarks | [code](code_examples/tensorflow/kernel_benchmarks) |

### Application examples

The `applications/` folder contains example applications written in different frameworks targeting the IPU. See the READMEs in each folder for details on how to use these applications.

### Code examples

The `code_examples/` folder contains small code examples showing you how to use various software features when developing for IPUs. See the READMEs in each folder for details.

### Tutorials

The `tutorials/` folder contains tutorials to help you get started using the Graphcore tools. Currently, this contains:

* `tutorials/poplar` - A set of tutorials to introduce the Poplar framework and the Poplibs libraries.

### Utilities

The `utils/` folder contains utilities libraries and scripts that are used across the other code examples. Currently this is split into:

* `utils/tests` - Common Python helper functions for the repo's unit tests.
* `utils/benchmarks` - Common Python helper functions for running benchmarks on the IPU in different frameworks.
