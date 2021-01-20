Keras tutorial: How to run on IPU
-----------------------------------

This tutorial provides an introduction on how to run Keras models on IPUs, and features that allow you to fully utilise the capability of the IPU. Please refer to the [TensorFlow 2 Keras API](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#module-tensorflow.python.ipu.keras) for further details.

Requirements:
* Installed and enabled Poplar
* Installed the Graphcore port of TensorFlow 2

Refer to the Getting Started guide for your IPU System for instructions.

#### File Structure

* `completed_example` A completed example of running Keras models on the IPU
* `demo.py` A demonstration script, where code is edited to illustrate the differences between running a Keras model on the CPU and IPU
* `README.md` This file
* `test` A directory that contains test scripts

#### Keras MNIST example


`demo.py` illustrates a simple example using MNIST dataset, which consists of 60000 training and 10000 test images of handwritten digits for classification task. MNIST classification is a toy example problem, but is sufficient to outline the concepts introduced in this tutorial.

Without changes, `demo.py` will run the Keras model on the CPU. It is based on the original Keras tutorial and as such is vanilla Keras code. You can run this now to see its output. This tutorial README will later show the modifications required to make it run on an IPU.

Running `python3 demo.py` gives the following output:

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0
_________________________________________________________________
dense (Dense)                (None, 10)                16010
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________

Training
Train on 60000 samples
Epoch 1/3
60000/60000 [==============================] - 11s 175us/sample - loss: 1.0647 - accuracy: 0.6661
Epoch 2/3
60000/60000 [==============================] - 10s 169us/sample - loss: 0.3137 - accuracy: 0.9051
Epoch 3/3
60000/60000 [==============================] - 10s 164us/sample - loss: 0.2274 - accuracy: 0.9324

Evaluation
10000/10000 [==============================] - 1s 131us/sample - loss: 0.1380 - accuracy: 0.9598
```

#### Running example on IPU

##### 1. Import necessary libraries

First, import the necessary libraries.

Add the following import statements to the beginning of `demo.py`:

```
from tensorflow.python.ipu import keras as ipu_keras
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import utils as ipu_utils
```

##### 2. Add IPU configuration

To use the IPU, you must create an IPU session configuration.

Add the following code after the model function definition in `demo.py`:

```
ipu_config = ipu_utils.create_ipu_config()
ipu_config = ipu_utils.auto_select_ipus(ipu_config, num_ipus=2)
ipu_utils.configure_ipu_system(ipu_config)
```

By specifying `num_ipus=2` in the configuration, a single TensorFlow virtual device is created in control of 2 IPUs. Since our simple Keras model only needs 1 IPU, data replication is automatically enabled; the same model is run on both IPUs with different batches of data. Note that the system will automatically select IPU MultiDevices that contain the number of IPUs we request. To see all available IPU MultiDevices, use the [`gc-info` command line tool](https://docs.graphcore.ai/projects/command-line-tools/en/latest/gc-info_main.html).

##### 3. Specify IPU strategy

Next, add the following code after the configuration:

```
# Create an execution strategy.
strategy = ipu_strategy.IPUStrategy()
```

The `tf.distribute.Strategy` is an API to distribute training across multiple devices. `IPUStrategy` is a subclass which targets a system with one or more IPUs attached. Another subclass, [IPUMultiWorkerStrategy](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#tensorflow.python.ipu.ipu_multi_worker_strategy.IPUMultiWorkerStrategy), targets a multiple system configuration.

##### 4. Wrap the model within the IPU strategy scope

Creating variables and Keras models within the scope of the `IPUStrategy` will ensure that they are placed on the IPU. However, the initialisation of variables will be performed on the CPU device.

Replace the following code

```
# Model.__init__ takes two required arguments, inputs and outputs.
model = keras.Model(*model_fn())

# Compile our model with Stochastic Gradient Descent as an optimizer
# and Categorical Cross Entropy as a loss.
model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"])

model.summary()
print('\nTraining')
model.fit(x_train, y_train, epochs=3, batch_size=64)
print('\nEvaluation')
model.evaluate(x_test, y_test)
```

with

```
with strategy.scope():
    ipu_model = ipu_keras.Model(*model_fn())
    # Compile our model as with the CPU example.
    ipu_model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"])
    ipu_model.summary()
    print('\nTraining')
    ipu_model.fit(x_train, y_train, epochs=3, batch_size=64)
    print('\nEvaluation')
    ipu_eval_res = ipu_model.evaluate(x_test, y_test, batch_size=64)
    print(f'loss: {ipu_eval_res[0]:.4f} - accuracy: {ipu_eval_res[1]:.4f}')
```
Note that the function `model_fn()` defined in `demo.py` can be readily reused. However, the model is now an instance of the `ipu.keras.Model` class.

##### 5. Result of changes

Running `python3 demo.py` now gives the following output:

```
Model: "ipu_model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0
_________________________________________________________________
dense (Dense)                (None, 10)                16010
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________

Training
Epoch 1/3
2020-11-23 14:41:00.454751: W tensorflow/compiler/tf2xla/kernels/random_ops.cc:52] Warning: Using tf.random.uniform with XLA compilation will ignore seeds; consider using tf.random.stateless_uniform instead if reproducible behavior is desired.
2020-11-23 14:41:00.503607: I tensorflow/compiler/plugin/poplar/driver/poplar_compiler.cc:548] Automatically replicating the TensorFlow model by a factor of 2.
468/468 [==============================] - 64s 137ms/step - loss: 0.9864 - accuracy: 0.6923
Epoch 2/3
468/468 [==============================] - 0s 411us/step - loss: 0.2962 - accuracy: 0.9078
Epoch 3/3
468/468 [==============================] - 0s 438us/step - loss: 0.2181 - accuracy: 0.9333

Evaluation
2020-11-23 14:42:34.748838: I tensorflow/compiler/plugin/poplar/driver/poplar_compiler.cc:548] Automatically replicating the TensorFlow model by a factor of 2
loss: 0.1336 - accuracy: 0.9587
```

Note that the training time has been significantly reduced by use of the IPU.

#### Pipelining


Pipelining can also be enabled to shard a Keras model across multiple IPUs. See [keras.PipelineModel](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html#tensorflow.python.ipu.keras.PipelineModel) for more details.

A pipelined model will execute multiple sections (stages) of a model on individual IPUs concurrently by pipelining mini-batches of data through the stages. When defining the graph structure, you can control what parts of the model go into which stages with the `PipelineStage` context manager.

Replace the model implementation in `demo.py` with:

```
def pipeline_model_fn():
    # Input layer - "entry point" / "source vertex".
    input_layer = keras.Input(shape=input_shape)

    # Add graph nodes for the first pipeline stage.
    with ipu_keras.PipelineStage(0):
        x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)

    # Add graph nodes for the second pipeline stage.
    with ipu_keras.PipelineStage(1):
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x
```

Any operations created inside a `PipelineStage(x)` context manager will be placed in the `x`th pipeline stage. Here, the model has been divided into two pipeline stages that run concurrently.

The staged model should now be an instance of the `PipelineModel` class rather than the `Model` class. Replace

```
ipu_model = ipu_keras.Model(*model_fn(input_shape, num_classes))
```

with

```
ipu_model = ipu_keras.PipelineModel(*pipeline_model_fn(input_shape, num_classes), gradient_accumulation_count=8)
```

The `gradient_accumulation_count` argument specifies the number of mini-batches that flow through the pipeline before the weights are updated - the intermediate gradients for each mini-batch are accumulated during pipeline execution.
Note that now the Keras Model requires 2 IPUs to run (since it is a two-stage pipeline), all of the configured IPUs are being used, therefore no automatic data parallelism is done.

Running `python3 demo.py` now gives the following output:

```
Model: "pipeline_model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0
_________________________________________________________________
dense (Dense)                (None, 10)                16010
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________

Training
Epoch 1/3
2020-12-10 11:01:17.413415: W tensorflow/compiler/tf2xla/kernels/random_ops.cc:52] Warning: Using tf.random.uniform with XLA compilation will ignore seeds; consider using tf.random.stateless_uniform instead if reproducible behavior is desired.
117/117 [==============================] - 68s 581ms/step - loss: 1.1178 - accuracy: 0.6416
Epoch 2/3
117/117 [==============================] - 0s 2ms/step - loss: 0.3268 - accuracy: 0.8982
Epoch 3/3
117/117 [==============================] - 0s 2ms/step - loss: 0.2291 - accuracy: 0.9308

Evaluation
loss: 0.1350 - accuracy: 0.9531
```

#### Completed example

The folder `completed_example` contains a complete implementation of the illustrated Keras model. Run `python3 completed_example/main.py` to run the standard Keras model on a CPU.

The `--use-ipu` and `--pipelining` flags allow you to run the Keras model on the IPU and (optionally) adopt the pipelining feature respectively. The gradient accumulation count can be adjusted with the `--gradient-accumulation-count` flag.

Note that the code in `completed_example` has been refactored into 3 parts:

* `main.py`: Main code to be run.

* `model.py`: Implementation of a standard Keras model and a pipelined Keras model.

* `utils.py`: Contains functions that load the data and argument parser.

#### License

This example is licensed under the Apache License 2.0 - see the LICENSE file in this directory.

Copyright (c) 2021 Graphcore Ltd. All rights reserved.

This directory contains derived work from the following:

Keras simple MNIST convnet example: https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py

Copyright holder unknown (author: François Chollet 2015)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.
