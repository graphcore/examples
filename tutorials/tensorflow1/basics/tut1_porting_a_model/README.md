# Tutorial 1: Porting a simple example

## Introduction

Porting code to the IPU is designed to be as simple as possible, and doing so only requires a few extra lines of code. We start with the basic import statements:

```python
# SNIPPET 1

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

# We'll be interested in performance later
import time
```

For the `ipu` module to function properly, we must import it directly rather than accessing it through the top-level TensorFlow module.

To provide some scaffolding for our explanations, we will create a simple convolutional model consisting of two convolutions followed by a dense layer. We will then train this model on the Fashion-MNIST dataset, using sparse softmax cross-entropy for our loss function and the Adam optimiser. After each epoch, we will report the average loss.

In this tutorial, we will use Keras to build our model and the `tf.Session` API to run our model, though an `Estimator` interface is also available.  We prepare a model for running within a `tf.Session` as we normally would, using the `tf.data` API to build an iterator over the data:


```python
# SNIPPET 2

# Specify hyperparameters for later use
BATCHSIZE = 32
EPOCHS = 5

# Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), _ = fashion_mnist.load_data()

# Cast and normalize the training data
x_train = x_train.astype('float32') / 255
y_train = y_train.astype('int32')

# Build iterator over the data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().batch(BATCHSIZE, drop_remainder=True)
dataset_iterator = dataset.make_initializable_iterator()

# Define the layers to use in model

# Fashion-MNIST images are greyscale, so we add a channels dimension
expand_dims = tf.keras.layers.Reshape((28, 28, 1))

conv = (tf.keras.layers.Conv2D(filters=8,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu,
                               data_format="channels_last"))

flatten = tf.keras.layers.Flatten()

final_dense = tf.keras.layers.Dense(10)

# Define the model

model = tf.keras.Sequential([expand_dims, conv, conv, flatten, final_dense])

# Use floor division because we drop the remainder
batches_per_epoch = len(x_train) // BATCHSIZE
```

We make two small IPU-specific adjustments here. Since the features are originally of type `uint8`, which is not supported on the IPU (see the relevant part of the [documentation](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/device_selection.html#supported-types), we cast them to `float32`. Also, we use `drop_remainder=True` when batching the data. This is because the dimensions of the tensors in a computational graph which is run on the IPU must be fixed.

## IPU Configuration

The next step is to configure the IPU device(s) we wish to run the model on. An "IPU device" in this case refers to a set of one or more IPUs. This can be done in just a few lines of code:

```python
# SNIPPET 3

# Create a default configuration
ipu_configuration = ipu.utils.create_ipu_config()

# Select an IPU automatically
ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)

# Apply the configuration
ipu.utils.configure_ipu_system(config=ipu_configuration)
```

This will create one TensorFlow virtual device which we can refer to later as `/device:IPU:0`. This virtual device will be able to use 1 IPU. We can create multiple TensorFlow virtual devices, each with multiple IPUs, by passing a Python list of sizes to `auto_select_ipus`. However, working effectively with multiple IPUs is a subject that requires separate treatment and will be covered in a further tutorial.

For many small applications, this is enough to get a model up and running, but there are still plenty of useful options. For example, in `create_ipu_config`, we can enable profiling to get information on compilation, execution and memory usage, and we can also target a specific IPU or multi-IPU device using `ipu.utils.select_ipus`. The full list of options is documented in the relevant part of the [API reference](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/api.html#tensorflow.python.ipu.utils.create_ipu_config).

## Preparing the model for the IPU

The next step is to prepare the model for running on the IPU. This is handled by the `ipu.ipu_compiler.compile` function, which takes a Python function and some inputs, and returns a TensorFlow operation which applies the computation to the inputs and can be run on the IPU using a `tf.Session`. To use it, we need to define a function which executes the main training loop:

```python
# SNIPPET 4

def training_loop_body(x, y):

    logits = model(x, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)

    return([loss, train_op])
```

Remember that one of our objectives is to report the average loss after each epoch, so we need our function to output the loss at each step. Because TensorFlow 1 uses lazy evaluation, we return `train_op` (the actual value of which is `None`) as well to ensure the training step is executed.

Now we can build an executable TensorFlow operation from the loop function. As stated above, this step is handled by `ipu.ipu_compiler.compile`. This takes a Python function `computation` and a list of inputs `inputs` and returns an operation which applies the computation to the inputs and can be run on an IPU device using a `sess.run` call. These inputs can be constants, `tf.placeholder` variables, or values from a dataset iterator. If we wish to pass inputs from a dataset iterator, we pass them from the `get_next()` method of the iterator.

When operations are built, they must be placed on a particular IPU device, as created in the configuration step earlier. This is done with the `ipu.scopes.ipu_scope` API, as demonstrated below:


```python
# SNIPPET 5

# Get inputs from get_next() method of iterator
(x, y) = dataset_iterator.get_next()

# We build the operation within the scope of a particular device
with ipu.scopes.ipu_scope('/device:IPU:0'):

    # Pass the training loop function and list of inputs to ipu.ipu_compiler.compile
    training_loop_body_on_ipu = ipu.ipu_compiler.compile(computation=training_loop_body, inputs=[x, y])
```

## Running the model

We can now run our training loop on an IPU using a `tf.Session`, with no further IPU-specific code required:


```python
# SNIPPET 6

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(dataset_iterator.initializer)

    for i in range(EPOCHS):

        loss_running_total = 0.0

        epoch_start_time = time.time()

        for j in range(batches_per_epoch):

            # Having been placed on the IPU, this part runs on the IPU
            loss = sess.run(training_loop_body_on_ipu)

            loss_running_total += loss[0]

        # Print average loss and time taken for epoch
        print('\n', end='')
        print("Loss:", loss_running_total/batches_per_epoch)
        print("Time:", time.time() - epoch_start_time)

print("Program ran successfully")
```

Under the hood, the `sess.run` call uses TensorFlow's XLA optimising compiler to generate an efficient version of the model, which is then converted into a Poplar executable and run on the IPU. You can read more about XLA [here](https://www.tensorflow.org/xla). The first run takes much longer than the others because the model must be compiled the first time it is run.

This completes the code example. As we have seen, the changes required to run code on an IPU are minimal, and in many cases (such as configuration) we can use ready-made pieces of code to do the work.

While this code does run on the IPU, it is not as fast as it could be, as there is significant overhead from calling `sess.run` every time we want to execute the training loop. Eliminating this overhead is covered in the second tutorial in this series.
