# Graphcore code examples

This repository contains sample applications and code examples for use with
Graphcore IPUs.

If you are interested in finding out more about Graphcore, including
getting preview access to IPUs to run these examples, please register
your interest [here](https://www.graphcore.ai/product_info).

Please note we are not currently accepting pull requests or issues on this
repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

## Repository contents

### Notable examples

| Example | Link |
| ------- | ---- |
| BERT | [code](applications/popart/bert) |
| ResNext | [code](applications/popart/resnext_inference) |
| EfficientNet | [code](applications/tensorflow/cnns/inference) |
| Recommendation using Autoencoders | [code](applications/tensorflow/autoencoder) |
| Sales forecasting example | [code](applications/tensorflow/sales_forecasting) |
| MCMC methods example | [code](code_examples/tensorflow/mcmc) |
| Recurrent layer kernel benchmarks | [code](code_examples/tensorflow/kernel_benchmarks) |
| ResNet | [inference code](applications/tensorflow/cnns/inference), [training code](applications/tensorflow/cnns/training) |
| Constrastive Divergence VAE using MCMC methods | [code](applications/tensorflow/contrastive_divergence_vae) |
| Grouped convolution kernel benchmarks | [code](code_examples/tensorflow/kernel_benchmarks) |
| Example reinforcement learning policy model | [code](applications/tensorflow/reinforcement_learning) |

### Application examples

The `applications/` folder contains example applications written in different frameworks targeting the IPU. See the READMEs in each folder for details on how to use these applications.

### Code examples

The `code_examples/` folder contains small code examples showing you how to use various software features when developing for IPUs. See the READMEs in each folder for details.

### Utilities

The `utils/` folder contains utilities libraries and scripts that are used across the other code examples. Current this is split into:

* `utils/tests` - Common python helper functions for the repo's unit tests.
* `utils/benchmarks` - Common python helper functions for running benchmarks on the IPU in different frameworks.
