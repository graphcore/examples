Graphcore: IPUEstimator CNN example
===
This README describes how to run the IPUEstimator to train and evaluate a simple CNN.


## Overview

TensorFlow Estimators use a high-level TensorFlow API, which automatically handles most of the implementation details when training/evaluating a model. They are designed to be "hardware agnostic"; you do not have to change your model when running on CPUs, GPUs, TPUs or over single devices versus distributed devices. For more information about Estimators, please see https://www.tensorflow.org/guide/estimator.

## IPUEstimator

The IPUEstimator is the IPU implementation of the TensorFlow Estimator. It handles several parts of running a TensorFlow program on the IPU for you, including:

1. Scoping the model inside the `ipu.scopes.ipu_scope('/device:IPU:0')` scope;
2. Compiling the model with XLA via `ipu.ipu_compiler.compile` with an `ipu.loops.repeat` loop;
3. Creating infeeds from the datasets and, if enabled, outfeeds;
4. Configuring the IPU system;
5. Incrementing the global step counter;
6. If requested, turning replication and sharding on;
7. Generating compilation/execution reports, etc.

Normally, you give an Estimator a RunConfig to tell the Estimator how to train/evaluate the model. We extend the RunConfig class to accept an IPURunConfig, which additionally accepts an IPU config, to allow you to express the IPU-specific options you would like, while still being able to use that RunConfig with another Estimator. In these configs, you can define things such as the iterations per loop on the IPU, the replication factor, autosharding, whether or not to generate a compilation report etc.

In this example, the IPUEstimator is used to train and evaluate a simple CNN based on https://keras.io/examples/cifar10_cnn/.

## Dataset

This model uses the standard CIFAR10 Keras dataset with contains 60,000 32x32 colour images in 10 classes, with 6000 images per class. There are 50,000 training images and 10,000 test images. It is automatically downloaded by the script.

## File structure

| IPUEstimator files     | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `README.md`            | This file, describing how to run the model                 |
| `ipu_estimator_cnn.py` | Main Python script to run the IPUEstimator for a CNN model |

## Quick start guide

1. Prepare the TensorFlow environment.
   Install the poplar-sdk following the README provided. Make sure to run the enable.sh scripts and activate a Python3 virtualenv with gc_tensorflow installed.
2. Train and test the model:

       python ipu_estimator_cnn.py

### Examples

To train the model with batch size 16, for 500 batches per IPU loop, with a learning rate of 0.001:

       python ipu_estimator_cnn.py --batch-size 16 --batches-per-step 500 --learning-rate 0.001

To test the model, loading weights from a checkpoint in a directory /path/to/ckpt:

       python ipu_estimator_cnn.py --test-only --model-dir /path/to/ckpt

To generate a compilation and execution report into a directory /path/to/reports:

       python ipu_estimator_cnn.py --profile --model-dir /path/to/reports

#### Options

The `ipu_estimator_cnn.py` script has a few options. Use the `-h` flag or examine the code to understand them.
