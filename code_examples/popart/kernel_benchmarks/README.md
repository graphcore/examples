# Synthetic benchmarks on IPUs

This readme describes how to run synthetic benchmarks for models with a single type of layer and synthetic data in training and inference.

## Overview

This example uses an LSTM model for benchmarking. LSTM (Long Short-Term Memory) is used in sequential data with long dependencies.

## Quick start guide

1. Prepare the environment. Install the Poplar SDK following the the instructions
   in the Getting Started guide for your IPU system. Make sure to run the `enable.sh` scripts.
2. Run the training program. For example:
   `python3 lstm.py`
   Use `--help` to show all available options.

## File structure

|            |                           |
|------------|---------------------------|
| `lstm.py`          | Benchmark program for 1 LSTM layer                       |


----

