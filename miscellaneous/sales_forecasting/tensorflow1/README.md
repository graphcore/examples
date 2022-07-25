# Graphcore example applications: sales forecasting
This README describes how to run a sales forecasting machine learning model on Graphcore's IPUs.

## Overview

This directory contains code to train a simple multi-layer perceptron (MLP) network to predict sales data.

The model predicts the amount of sales on a particular day given a set of features in the original Rossmann competition dataset.

## Running the model

The following files are provided for running the sales forecasting model.

* `README.md`   This file.
* `main.py`     Main training and validation loop.
* `data.py`     Data pipeline.
* `model.py`    Model file.
* `util.py`     Helper functions.
* `test_model.py`   A test script. See below for how to run it.

## Quick start guide

### Prepare the environment

**1) Download the Poplar SDK**

Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. 
Make sure to source the `enable.sh` scripts for poplar.

**2) Python**

Create a virtualenv and install the required packages:

```
bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install <path to the tensorflow-1 wheel file from the Poplar SDK>
pip install -r requirements.txt
```

### Prepare dataset

**1) Download dataset from Kaggle**

The data for this example is from a Kaggle competition. You will need to create a Kaggle account from here: https://www.kaggle.com/account/login. Then, navigate to https://www.kaggle.com/c/rossmann-store-sales/data) and press the "Download All" button. If you haven't already, you will be asked to verify your Kaggle account via your mobile phone. After entering a valid mobile phone number you will receive an SMS message with a verification code. Alternatively, you can use the Kaggle API (https://github.com/Kaggle/kaggle-api) to download it via command line with `kaggle competitions download rossmann-store-sales` after setting up your Kaggle API token.

**2) Extract the data into a folder**

For example:

    unzip rossmann-store-sales.zip -d rossmann-data

### Run the training program

Run the program using `main.py`. Use the `--datafolder`/`-d` option to specify the path to the data folder. For example:

    python main.py -d rossmann-data

### Options

Use `--help` to show the available options.

`--replication-factor` will add data parallelism. IPU graph replication copies the graph to N IPUs, splits the data into N streams and trains the N graphs on the N streams in parallel. Periodically, the graphs' gradients are averaged and applied to all graphs. Set the number of replicas with this option. By default, no replication is done (i.e. this is 1). If validating with the `--multiprocessing` flag, this can be at max M/2, where M is the number of IPUs available, as each process needs N IPUs.

`--multiprocessing` is recommended if at least 2 IPUs are available. It will run the training and validation graphs in separate processes on separate IPUs, saving time loading and unloading programs from the device. By default, the same IPUs are shared for both training and validation, executed in sequence.

`--no-validation` disables validation.

`--lr-schedule-type`: The model can use a manual or a dynamic learning rate schedule.
For _manual_ learning rate schedules, you can specify _n_ change points, as a ratio of training progress, and _n+1_ values of learning rate at each interval.
For example, we can specify `learning-rate-schedule` as a comma separated list of two values `0.33,0.66` and `learning-rate-decay` as a comma separated list of three values `1,0.1,0.01` to drop the learning rate by a factor of 10 at 33% of training, and by a factor of 100 at 66% of training (relative to the initial learning rate).

For _dynamic_ learning rate schedules, the model attempts to dynamically update learning rate based on validation (or training, if validation isn't supplied) progress.
The dynamic scheduler has two features:
* Reduction on plateau - where learning rate is reduced if training is stagnating. Set `--lr-plateau-patience` and `--lr-schedule-plateau-factor` for the responsiveness to training stagnation and rate of reduction of this mechanism.
* A LR warmup at the start of training, off by default - where learning rate is gradually increased to its initial value at the start of training over a number of epochs. Set `--lr-warmup` to enable this mechanism and `--lr-warmup-epochs` to set the number of epochs the learning rate is warmed up over. The former mechanism doesn't apply when learning rate is being warmed up.

`--base-learning-rate` specifies the exponent of the base learning rate. The learning rate is set by `lr = 2^blr * micro-batch-size`. See <https://arxiv.org/abs/1804.07612> for more details.

`--d`/`--datafolder` sets the directory of the data folder. This should contain a preprocessed Rossmann dataset of `train.csv` and `val.csv`. Alternatively use `--use-synthetic-data` to use random data generated directly on the IPU as needed by the program, removing any host <-> IPU data transfers.

`--log-dir` sets the directory for model logs. The model will log summaries and checkpoints.

`--micro-batch-size` and `--validation-batch-size` set the batch sizes of the training and validation graphs respectively.

`--precision` sets the precision of the variables and calculations respectively. Supply as a dot separated list e.g. `32.32` or `16.16`

`--no-prng` disables stochastic rounding.

`--loss-scaling` sets the scaling of the loss. By default, this is 1 i.e. no loss scaling is done.

`--weight-decay` sets the rate of decay of the dense layers' kernels in the model.

`--gradient-clipping` clips the gradients between -1 and 1 before being applied in each update step.

`--epochs` sets the number of epochs to train for.

`--select-ipus` selects the IPUs to use. Use `AUTO` for automatic selection from available IPUs. Use a comma separated list of IPU IDs to specify the IPUs, by ID, for the training and validation processes to run on, respectively.

`--valid-per-epoch` sets the number of times to validate per epoch.

`--device-iterations` sets the number of global batches / weight updates to complete every training step.

`--steps-per-log` sets the number of steps to take before logging, to output, of current training progress, repeatedly.

`--use-init` sets the initial weights of the model to be the same across separate runs.

`--compile-only` causes the model to only compile. This will not acquire any IPUs and thus facilitates
profiling without using hardware resources.

`--compile-only-ipu-version` sets the IPU version to be used while using --compile-only option

### Test script

The test script performs basic tests. To run it you need add the `utils` directory at the top-level of this repository
to your `PYTHONPATH` and also run `pip install pytest`. Then run

    pytest test_main.py

## Benchmarking

To reproduce the benchmarks, please follow the setup instructions in this README to setup the environment, and then from this dir, use the `examples_utils` module to run one or more benchmarks. For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml
```

or to run a specific benchmark in the `benchmarks.yml` file provided:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --benchmark <benchmark_name>
```

For more information on how to use the examples_utils benchmark functionality, please see the <a>benchmarking readme<a href=<https://github.com/graphcore/examples-utils/tree/master/examples_utils/benchmarks>

## Profiling

Profiling can be done easily via the `examples_utils` module, simply by adding the `--profile` argument when using the `benchmark` submodule (see the <strong>Benchmarking</strong> section above for further details on use). For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --profile
```
Will create folders containing popvision profiles in this applications root directory (where the benchmark has to be run from), each folder ending with "_profile". 

The `--profile` argument works by allowing the `examples_utils` module to update the `POPLAR_ENGINE_OPTIONS` environment variable in the environment the benchmark is being run in, by setting:
```
POPLAR_ENGINE_OPTIONS = {
    "autoReport.all": "true",
    "autoReport.directory": <current_working_directory>,
    "autoReport.outputSerializedGraph": "false",
}
```
Which can also be done manually by exporting this variable in the benchmarking environment, if custom options are needed for this variable.
