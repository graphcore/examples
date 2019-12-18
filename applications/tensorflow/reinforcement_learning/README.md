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

### Dataset

The function `env_generator` simulates discrete and continuous observations along with simulated rewards under the current policy.

## Running the model

The following files are included in this repo:

| File              | Description               |
| ----------------- | ------------------------- |
| `README.md`       | How to run the model      |
| `rl_benchmark.py` | The main training program |


## Quick start guide

1. Prepare the TensorFlow environment.
   Install the poplar-sdk following the README provided. Make sure to run the enable.sh scripts and activate a Python virtualenv with gc_tensorflow installed.
2. Run the training program.
   `python3 rl_benchmark.py --batch_size 8 --time_steps 16 --num_ipus 8`

Use `--help` to show all available options.

