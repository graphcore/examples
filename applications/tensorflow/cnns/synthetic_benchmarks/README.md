# Synthetic benchmarks on IPUs

This readme describes how to run synthetic benchmarks for models such as ResNet on the IPU.

## Overview

Deep CNN residual learning models such as ResNet are used for image recognition and classification. The synthetic benchmark 
example uses a ResNet model implemented in TensorFlow, optimised for Graphcore's IPU.

## Quick start guide

1. Prepare the TensorFlow environment. Install the poplar-sdk following the README provided. Make sure to run the 
   `enable.sh` scripts and activate a Python virtualenv with gc_tensorflow installed.
2. Run the training program. For example: 
   `python3 resnet.py`
   Use `--help` to show all available options.

## File structure

|            |                           |
|------------|---------------------------|
| `resnet.py`      | The main training program |


----

