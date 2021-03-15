# Octave Convolutions in PopTorch

## Overview

This example shows how to use PopTorch training and inference models that use
Octave Convolutions as described in the paper:
["Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution"](https://arxiv.org/pdf/1904.05049.pdf)

## Installation

1. Prepare the PopTorch environment. Install the Poplar SDK following the
   Getting Started guide for your IPU system. Make sure to source the
   `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv with PopTorch installed.
2. Install additional Python packages specified in requirements.txt
```:bash
pip3 install -r requirements
```
This will install an [octconv implementation](https://github.com/braincreators/octconv) for PyTorch from GitHub.

## Execution

This [example](octconv_example.py) implements a simple image classification model.
This full example uses PopTorch to train the model and then evaluate it on the IPU with the CIFAR-10 dataset.
The classification model can be configured to use one of the following convolution modes:

* `vanilla`: the standard `torch.nn.Conv2d` implementation is used for all convolutions.
* `octave`: use the Octave convolution `octconv.OctConv2d` in place of all convolutions. 
* `multi-octave`: uses `poptorch.MultiConv` to execute the data-parallel convolutions that make-up an Octave convolution in parallel.

The Octave convolutions are parameterized by the `alpha` value which defines the ratio of low-frequency maps.

These options can be selected from the command-line:
```:bash
python3 octconv_example.py --conv-mode octconv --alpha 0.8
```

## Full Usage

```
usage: octconv_example.py [-h] [--conv-mode {vanilla,octave,multi-octave}]
                          [--alpha ALPHA] [--batch-size BATCH_SIZE]
                          [--batches-per-step BATCHES_PER_STEP]
                          [--test-batch-size TEST_BATCH_SIZE]
                          [--epochs EPOCHS] [--lr LR]
                          [--profile-dir PROFILE_DIR] [--cache-dir CACHE_DIR]
                          [--data-dir DATA_DIR] [--expansion EXPANSION]

Octave Convolution in PopTorch

optional arguments:
  -h, --help            show this help message and exit
  --conv-mode {vanilla,octave,multi-octave}
                        Convolution implementation used in the classification
                        model (default: vanilla)
  --alpha ALPHA         Ratio of low-frequency features used in octave
                        convolutions (default: 0.5)
  --batch-size BATCH_SIZE
                        batch size for training (default: 8)
  --batches-per-step BATCHES_PER_STEP
                        device iteration (default:50)
  --test-batch-size TEST_BATCH_SIZE
                        batch size for testing (default: 80)
  --epochs EPOCHS       number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.05)
  --profile-dir PROFILE_DIR
                        Perform a single iteration of training for profiling
                        and place in specified folder.
  --cache-dir CACHE_DIR
                        Enable executable caching in the specified folder
  --data-dir DATA_DIR   Location to use for loading the CIFAR-10 dataset from.
  --expansion EXPANSION
                        Expansion factor for tuning model width.
```