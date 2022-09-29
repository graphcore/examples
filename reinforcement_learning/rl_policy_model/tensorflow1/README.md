# Graphcore benchmarks: Reinforcement Learning

This readme describes how to train a deep reinforcement learning model on multiple IPUs with synchronous data parallel training using synthetic data.

## Overview

The general goal with Reinforcement Learning (RL) is to maximise some long-term reward by mapping observations and measurements to a set of actions. This usually involves an agent of some kind learning an optimal sequence of decisions. Useful applications of RL are therefore in areas where automated sequential decision-making is required.

Deep reinforcement learning combines the strength of deep neural networks (learning useful features from observations) with the machine learning paradigm of learning from trial and error. Graphcore has run a deep reinforcement learning model (policy gradient model) on multiple IPUs with synchronous data parallel training using synthetic data.

The model contains the following layers which are typically found in a policy network:

- Embedding layers for representing discrete observations
- Clipping layers to clip the values of continuous observations
- Concatenating layers to group observations
- Fully-connected transformations
- Layers for choosing maximum feature value along a specific dimension
- LSTM layer to process a sequence of observations
- Final softmax layer of size = num_actions

## Quick start guide

The following files are included in this repo:

| File              | Description               |
| ----------------- | ------------------------- |
| `README.md`       | How to run the model      |
| `rl_benchmark.py` | The main training program |
| `test_rl.py`      | Unit tests                |

### Prepare environment

1. Install the Poplar SDK following the the instructions in the Getting Started guide for your IPU system.
   Make sure to run the enable.sh script and activate a Python virtualenv with the tensorflow-1 and ipu_tensorflow_addons for tensorflow-1 wheels from the Poplar SDK installed.

```shell
virtualenv --python python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
pip install <path to the TensorFlow-1 wheel from the Poplar SDK>
pip install <path to the ipu_tensorflow_addons wheel for TensorFlow 1 from the Poplar SDK>
```

2. Run the training program.
   `python3 rl_benchmark.py --micro_batch_size 8 --time_steps 16 --num_ipus 8`

### Dataset

The function `env_generator` simulates discrete and continuous observations along with simulated rewards under the current policy.

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

### Running the model

Start training the model with:

```shell
python3 rl_benchmark.py --micro_batch_size 8 --time_steps 16 --num_ipus 8
```

Use `--help` to show all available options.
