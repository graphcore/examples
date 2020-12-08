## Graphcore benchmarks: UNet Industrial

This readme describes how to run UNet Industrial example with TensorFlow 1.15. This directory contains all the code required to do this on Graphcore's IPU.

## UNet Industrial Model Overview

UNet is an image segmentation model originally proposed by Ronneberger et. al for biomedical tasks: https://arxiv.org/abs/1505.04597
It was adapted later by NVIDIA for industrial use cases: https://ngc.nvidia.com/catalog/resources/nvidia:unet_industrial_for_tensorflow
The model is a convolutional autoencoder with additional skip connections between the encoder and decoder parts.

IMPORTANT: This example requires Mk2 IPU silicon to run. It will result in an error on Mk1 IPUs.

## Datasets

This example uses random data generated on the host to make sure that throughput numbers match those of real-world scenarios.

## Running the model

This repo contains the code required to run the UNet Industrial with TensorFlow 1.15

The structure of the repo is as follows:

| File                      | Description                          |
| ------------------------- | ------------------------------------ |
| `unet_industrial.py`      | Main Python script                   |
| `requirements.txt`        | Required Python modules and versions |
| `README.md`               | This file                            |
| `test_unet_industrial.py` | Unit test script                     |


## Quick start guide

### Prepare the environment

**1) Download the Poplar SDK**

Download the Poplar SDK version 1.3 or higher and follow the README provided to install the drivers, set up the environment, and install TensorFlow 1.15 for the IPU. Skip this step if you have done this already.

**2) Install required modules**

In this example's directory, run:

```
pip install -r requirements.txt
```

### Run the UNet Industrial program

Run as follows:

```
python3 unet_industrial.py
```

Sample output:

```
UNet Industrial example
..
Training...
..
Mean throughput: XXX images/sec
..
Evaluating...
..
Evaluation loss: XXX

Testing inference...
..
Throughput: XXX items/second

Completed
```

Note that the loss can be high due to the data being random.