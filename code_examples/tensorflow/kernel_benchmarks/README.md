## Graphcore benchmarks: Grouped convolution, GRU, LSTM and RNN layers with TensorFlow

This readme describes how to run the benchmarks for models with a single type of layer and synthetic data in training and inference.

## Benchmarking

To reproduce the published Mk2 throughput benchmarks, please follow the setup instructions in this README, and then follow the instructions in [README_Benchmarks.md](README_Benchmarks.md) 

## Overview

Each program creates a model with only one type of layer for benchmarking.
* Grouped convolutions are used in CNNs such as ResNeXt models.
* RNN (Recurrent Neural Network) are used in sequential data.
* LSTM (Long Short-Term Memory) are used in sequential data with long dependencies.
* GRU (Gated Recurrent Unit) are a simpler version of LSTM.
* Dense layers, also known as fully connected layers are widely used across a range of models.
* HMC The HMC step from Contrastive Divergence for Combining Variational Inference and MCMC.

## Running the model

This repo contains the code required to run the kernel model.

The structure of the repo is as follows:

| File                                            | Description			                                                       |
| ----------------------------------------------- | ---------------------------------------------------------              |
| `grouped_conv.py`                               | Benchmark program for grouped convolutions                             |
| `gru.py`                                        | Benchmark program for 1 GRU layer                                      |
| `hmc.py`                                        | Benchmark HMC steps in isolation                                       |
| `lstm.py`                                       | Benchmark program for 1 LSTM layer                                     |
| `lstm_multi.py`                                 | Benchmark for a multi-layer LSTM with a dense final layer              |
| `rnn.py`                                        | Benchmark program for 1 RNN layer                                      |
| `dense.py`                                      | Benchmark program for 1 Dense layer                                    |
| `README.md`                                     | This file                                                              |
| `test/`                                         | Test code that can be run via pytest                                   |
| `dynamic_sparse_transformer_encoder_layer.py`   | Benchmark program for 1 sparse tranformer encoder layer                |
| `dynamic_sparse_transformer_ffn_block.py`       | Benchmark program for 1 sparse tranformer forward block                |
| `dynamic_sparse_transformer_self_attention.py`  | Benchmark program for 1 sparse tranformer self attention layer layer   |
| `dynamic_sparse_gru.py`                         | Benchmark program for 1 sparse gru layer                               |
| `dynamic_sparse_fc.py`                          | Benchmark program for 1 sparse fully connected layer                   |



## Quick start guide

### Prepare the environment

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
  Make sure to source the `enable.sh` script for poplar, and activate a Python
  virtualenv with the tensorflow-1 wheel from the Poplar SDK installed.

### Run the programs

Each benchmark can be executed multiple times using the `--steps`
option.

```
python3 program.py --steps 5
```

Use `--help` or `-h` to see all available options.

