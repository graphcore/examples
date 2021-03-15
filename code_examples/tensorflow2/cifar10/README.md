# Graphcore

## CIFAR-10 Estimators

This example shows how to use IPU-specific Estimators as an alternative to the
IPU-specific Keras Model classes (see tutorials/tensorflow2/keras).

The starting point of this code example is the [Keras CIFAR-10 example](https://github.com/keras-team/keras/blob/1a3ee8441933fc007be6b2beb47af67998d50737/examples/cifar10_cnn.py).

### File structure

* `cifar10_ipu_estimator.py` The main Python script to run IPUEstimator.
* `cifar10_estimator_replica.py` The main Python script to run IPUEstimator with multiple replicas.
* `cifar10_pipeline_estimator.py` The main Python script to run IPUPipelineEstimator.
* `README.md` This file.
* `requirements.txt` Required packages for the tests
* `test_cifar10.py` Integration tests

### How to use this demo

1) Prepare the TensorFlow environment.

Install the Poplar SDK. Make sure to run the enable.sh scripts and activate a Python 3 virtualenv with the tensorflow-2 wheel from the Poplar SDK installed.

2) Train and test the model: start the selected file as described below.

#### Using IPUEstimator

Using IPUEstimator provides IPUInfeedQueue and IPUOutfeedQueue and has the interface of TensorFlow's `tf.Estimator`.
A few minor modifications are required compared to the previous version to make the model work:

* The data source must be generated-based `tf.data.DataSet`. This Dataset must be wrapped into a function before passing it to the train/eval function.
* A training function also must be created, which acts as the specification.

Run with the following command:

```bash
python3 cifar10_ipuestimator.py
```

#### Data-parallel training using replicas

Using multiple IPUs can increase further the throughput. In this modification, multiple IPUs are used in data-parallel way.

The following modifications are required:

* The optimizer must be wrapped in `CrossReplicaOptimizer`. Note that this only supports optimizers that are subclasses of `tensorflow.python.training.optimizer.Optimizer`.

Run with the following command:

```bash
python3 cifar10_replica.py
```

#### Using IPUPipelineEstimator

Another way of improving throughput is Pipelining.

The following modifications are required:

* The model must be sharded into multiple stages.
* IPUPipelineEstimator must be used instead of IPUEstimator

Run with the following command:

```bash
python3 cifar10_pipeline.py
```

### Tests

Install the required packages:

```bash
pip3 install -r requirements.txt
```

Run the tests:

```bash
python3 -m pytest
```

#### License
This example is licensed under the MIT license - see the LICENSE file at the top level of this repository.

It includes derived work from:

Keras, https://github.com/keras-team/keras/tree/1a3ee8441933fc007be6b2beb47af67998d50737
(Source file has been deleted from the master branch)

All contributions by François Chollet:
Copyright (c) 2015 - 2019, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2019, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2019, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2019, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
