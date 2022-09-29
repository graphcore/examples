## Graphcore benchmarks: MCMC with TFP

This readme describes how to run MCMC sampling examples with TFP, TensorFlow Probability. This directory contains all the code required to do this on Graphcore's IPU.

## Overview

TensorFlow Probability (TFP) is a Python library for probabilistic models built on TensorFlow. It contains a wide selection of probability distributions and bijectors with tools to build deep probabilistic models, including probabilistic layers and the Edward2 language. Optimizers such as Nelder-Mead, BFGS, and SGLD are included and it can be used with both variational inference and Markov Chain Monte Carlo (MCMC). TFP is open source and available on GitHub: https://github.com/tensorflow/probability, a guide to it can be found here: https://www.tensorflow.org/probability/overview.

Markov Chain Monte Carlo methods for probabilistic machine learning are well known techniques used to solve integration and optimisation problems in large dimensional spaces. MCMC methods sample from a probability distribution based on a Markov chain equilibrium distribution.

## MCMC model

This MCMC model has been implemented using TensorFlow Probability to explore the performance across hardware platforms. The model is a neural network with 3 fully connected layers. The input data are features, generated from a time series of stock prices. Distributions of model parameters are represented by their samples. The samples are obtained using the Hamiltonian Monte Carlo (HMC) algorithm, which is an MCMC method, efficient in high-dimensional cases.

## Datasets

This example uses a proprietary dataset representative of algorithmic trading scenarios. The dataset aligns 146 features with observed returns.

The dataset has Copyright (c) 2021 by Carmot Capital LLC and Graphcore Ltd. All rights reserved. The dataset is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
  
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, the dataset distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Running the model

This directory contains the code required to run the MCMC model with TFP.

The structure of this directory is as follows:

| File               | Description                           |
| ------------------ | ------------------------------------- |
| `mcmc_tfp.py`      | Main Python script                    |
| `get_data.sh`      | Data preparation helper script        |
| `requirements.txt` | Required Python packages and versions |
| `README.md`        | This file                             |



## Quick start guide

### Prepare the environment

**1) Install the Poplar SDK**

Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
Make sure to source the `enable.sh` script for Poplar and activate a Python virtualenv with the tensorflow-1 wheel from the Poplar SDK installed.

**2) Install required modules**

In this example's directory, run:

```
pip install -r requirements.txt
```

**3) Download the data**

Execute the following command:

```
sh get_data.sh
```

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

## Run the MCMC program

Run as follows:

```
python3 mcmc_tfp.py
```
