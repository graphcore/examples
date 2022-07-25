# Synthetic benchmarks on IPUs

This readme describes how to run synthetic benchmarks for models such as ResNet on the IPU.

## Overview

Deep CNN residual learning models such as ResNet are used for image recognition and classification. The synthetic benchmark
example uses a ResNet model implemented in TensorFlow, optimised for Graphcore's IPU.

## Quick start guide

1. Prepare the TensorFlow environment. Install the Poplar SDK following the instructions
   in the Getting Started guide for your IPU system. Make sure to run the
   `enable.sh` script and activate a Python virtualenv with tensorflow-1 wheel from the Poplar SDK installed.
2. Run the training program. For example:
   `python3 resnet.py`
   Use `--help` to show all available options.

## File structure

|            |                           |
|------------|---------------------------|
| `resnet.py`      | The main training program |


----

