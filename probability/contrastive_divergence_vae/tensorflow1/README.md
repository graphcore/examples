# Graphcore benchmarks: Contrastive Divergence Variational Autoencoder

This readme describes how to train a Variational Autoencoder/Markov Chain Monte Carlo hybrid model on IPUs. 

## Overview

The Variational Autoencoder model used in these benchmarks is based on a method combining VI (Variational Inference) and MCMC (Markov Chain Monte Carlo) with a *Variational Contrastive Divergence* (VCD). This model is taken from the Variational Contrastive Divergence paper by Ruiz and Titsias: https://arxiv.org/abs/1905.04062, presented at ICML 2019.

## Variational autoencoder model

Motivated by the novelty of the method and the unusual computational structure of the model, we wrote this implementation in TensorFlow to explore the performance across hardware platforms. 

## Datasets

The model is trained with the statically binarised MNIST dataset discussed by Salakhutdinov and Murray (https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf) and released by Hugo Larochelle. The data will be downloaded automatically when the code is run for the first time.

## Running the model

This repo contains the code used for the experiments in the blog (https://www.graphcore.ai/posts/probabilistic-modelling-by-combining-markov-chain-monte-carlo-and-variational-inference-with-ipus), which relate to the above model. All the TensorFlow needed to reproduce the results is included.

The structure of the repo is as follows:

- `experiments/` contains `generative.py` which includes the high-level experiment control flow, data-loading and preprocessing
- `models/` includes the code for training and evaluating models
- `utils/` has the auxiliary code, used by the classes in the above folders
- `machinable/` contains some additional functionality for saving results in a standardised form
- `configs/` includes the configuration files for the experiments. Some of the hyperparameters can be set via the command line, or the config files can be copied and overridden. See Options section below for more information.
- `main.py` includes config overrides and the call to run the experiment
- `requirements.txt` is a list of PyPI dependencies

## Quick start guide

### Prepare the environment

**1) Download the Poplar SDK**

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  script for poplar.

##### 2) Python

Create a virtualenv with python 3.6 and install the required packages:

```
bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is not included in the requirements. If running on IPU hardware install the Graphcore tensorflow wheel from the Poplar SDK as follows:

```
pip install path/to/tensorflow-<version>-cp36-cp36m-linux_x86_64.whl
```

Otherwise install standard TensorFlow 1.14.0.

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

### Run the training program

After activating your virtual environment, run as follows:

```
python3 main.py -r where/to/store_results/
```

The models defined by `configs/default_config.json` and `configs/global_cv_config.json` take around nine hours to train on CPU.

### Options

Use `--help` to show the available hyperparameters and options. The detailed configuration is set in the files within the `configs/` folder. Set the path to the desired config file using the `-c` option:  

```
python3 main.py -c path/to/config.json
```

The preset configs included in `configs/` are:

- `default_config.json`  which is the experimental set up used in the original paper
- `global_cv_config.json` which is the same as `default_config.json` but with a scalar (global) control variate in place of a vector of local control variates
- `bs_experiment_config.json` is for the experiments testing the effect of batch size. These run for 8,200 epochs rather than 800 (as in the two configs above). You'll need to pass the `--micro_batch_size` argument via the command line, and specificy a learning rate using `--learning_rate` if the batch size is not one we tested (16, 128, 512, 1024). If the learning rate is not given and the batch size is one we tested, the learning rate we found to be best during validation will be used (see the `TUNED_LEARNING_RATE` dict in `main.py`)
- `test_config.json`  which is similar to default config but only run for one epoch

You can also copy and extend your own config — just specify the path to your custom config via the command line.

### Device Placement

The code will execute on IPU if an IPU is available, else it will execute on GPU if `tf.test.is_gpu_available()` returns `True`. If neither IPU nor GPU are available, it will run on CPU. This logic is implemented in `get_device_config()` in `utils/ipu_utils.py`. Tests only on IPU and fail if no IPU is available on the system.

### Tests

After setting up the environment, tests can be run with:

```
pytest
```
