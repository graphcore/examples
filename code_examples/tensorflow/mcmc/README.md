## Graphcore benchmarks: MCMC with TFP 

This readme describes how to run MCMC sampling examples with TFP, TensorFlow Probability. This directory contains all the code required to do this on Graphcore's IPU.

## Overview

TensorFlow Probability (TFP) is a Python library for probabilistic models built on TensorFlow. It contains a wide selection of probability distributions and bijectors with tools to build deep probabilistic models, including probabilistic layers and the Edward2 language. Optimizers such as Nelder-Mead, BFGS, and SGLD are included and it can be used with both variational inference and Markov Chain Monte Carlo (MCMC). TFP is open source and available on GitHub: https://github.com/tensorflow/probability, a guide to it can be found here: https://www.tensorflow.org/probability/overview. 

Markov Chain Monte Carlo methods for probabilistic machine learning are well known techniques used to solve integration and optimisation problems in large dimensional spaces. MCMC methods sample from a probability distribution based on a Markov chain equilibrium distribution.  

## MCMC model

This MCMC model has been implemented using TensorFlow Probability to explore the performance across hardware platforms. The model is a neural network with 3 fully connected layers. The input data are features, generated from a time series of stock prices. Distributions of model parameters are represented by their samples. The samples are obtained using the Hamiltonian Monte Carlo (HMC) algorithm, which is an MCMC method, efficient in high-dimensional cases.

## Datasets

This example uses a proprietary dataset representative of algorithmic trading scenarios. The dataset aligns 146 features with observed returns.

## Running the model

This repo contains the code required to run the MCMC with TFP.

The structure of the repo is as follows:

| File               | Description                          |
| ------------------ | ------------------------------------ |
| `mcmc_tfp.py`      | Main Python script                   |
| `get_data.sh`      | Data preparation helper script       |
| `requirements.txt` | Required Python modules and versions |
| `README.md`        | This file                            |



## Quick start guide

### Prepare the environment

**1) Download the Poplar SDK**

Download the Poplar SDK and follow the README provided to install the drivers, set up the environment, and install TensorFlow for the IPU. Skip this step if you have done this already.

**2) Install required modules**

NumPy and TensorFlow Probability are required. Note that you need TensorFlow Probability version 0.7 to run with TensorFlow version 0.14. In this example's directory, run:

```
pip install -r requirements.txt
```

**3) Download the data**

Execute the following command:

```
sh get_data.sh
```

### Run the MCMC program

Run as follows:

```
python3 mcmc_tfp.py
```