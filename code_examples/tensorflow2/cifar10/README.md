# Graphcore

## CIFAR-10 Keras training example

This example shows four simple approaches to port a TensorFlow Keras model to the IPU.

The starting point of this code example is the [Keras CIFAR-10 example](https://keras.io/examples/cifar10_cnn/).

### File structure

* `cifar10_ipu_strategy.py` The Python script to run IPUStrategy.
* `cifar10_ipu_estimator.py` The main Python script to run IPUEstimator.
* `cifar10_estimator_replica.py` The main Python script to run IPUEstimator with multiple replicas.
* `cifar10_pipeline_estimator.py` The main Python script to run IPUPipelineEstimator.
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

Install the Poplar SDK. Make sure to run the enable.sh scripts and activate a Python virtualenv with the TensorFlow 2 gc_tensorflow-2* wheel installed.

2) Install the package requirements

   `pip install -r requirements.txt`

3) Train and test the model: start the selected file as described below.

### Use IPUStrategy scope

The entire model creation, compile, fit and evaluation is placed inside IPUStrategy scope, which makes it possible to run on IPU with as little code modification as possible.

Run the following command:

```bash
python3 cifar10_ipu_strategy.py
```

### Use IPUEstimator

Using IPUEstimator provides IPUInfeedQueue and IPUOutfeedQueue and has the interface of TensorFlow's tf.Estimator
A few minor modifications are required compared to the previous version to make the model work:

* The data source must be generated-based `tf.data.DataSet`. This Dataset must be wrapped into a function before passing it to the train/eval function.
* A training function also must be created, which acts as the specification.

Run with the following command:

```bash
python3 cifar10_ipuestimator.py
```

### Data-parallel training using replicas

Using multiple IPUs can increase further the throughput. In this modification, multiple IPUs are used in data-parallel way.

The following modifications are required:

* The optimizer must be wrapped into CrossReplicaOptimizer

Run with the following command:

```bash
python3 cifar10_replica.py
```

### Use IPUPipelineEstimator

Another way of improving throughput is Pipelining.

The following modifications are required:

* The model must be sharded to multiple parts.
* IPUPipelineEstimator must be used instead of IPUEstimator

Run with the following command:

```bash
python3 cifar10_pipeline.py
```
