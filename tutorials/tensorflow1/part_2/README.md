# Tutorial 2: Loops and data pipelines

## Introduction and overview

In the first tutorial in this series, we walked through the steps of getting a simple model to run on the IPU. However, the resulting code was not as fast as it could have been, as there was significant overhead from calling `sess.run` every time the training loop body was executed. To eliminate this overhead, we need a way to run a looped function inside a single `sess.run` call.

TensorFlow for IPU handles this with the `ipu.loops` API, which allows us to create on-device loops for the IPU. The function which forms the body of the loop takes its previous output and/or the next batch of data from a dataset as inputs. 

To create a dataset the loop can iterate over, a `tf.data.Dataset` object is wrapped in an IPU infeed queue, an object which adds IPU-specific infeed operations and serves as a glue between the dataset and the `ipu.loops` API.

Now that we have the big picture, we can start to look at the details of the code. The import statements, data loading/processing and model definition are exactly as in the first tutorial, except we will not need an iterator over the data:

```python
# SNIPPET 1

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
import time


# Specify hyperparameters for later use
BATCHSIZE = 32
EPOCHS = 5

# Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), _ = fashion_mnist.load_data()

# Normalize and cast the data
x_train = x_train.astype('float32') / 255
y_train = y_train.astype('int32')

# Create and configure the dataset for iteration
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().batch(BATCHSIZE, drop_remainder=True)

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

## Creating an infeed queue

In order to create a data pipeline to the IPU, we wrap our `tf.data.Dataset` in an infeed queue, which manages the transfer of the data from the host to the IPU. All queues (including outfeed queues, which are not covered here) also need a `feed_name`, and all queues must have a different value for `feed_name`.

```python
# SNIPPET 2

mnist_infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="mnist_data")
```

Once the infeed queue has been created, the dataset itself must not be changed or used. All access to the data must be through the infeed queue. The only thing left to do is initialize the infeed queue within the `tf.Session` where we run our code, as we would with a standard dataset iterator.


## Creating a loop

The next step is to create a looped version of our training function, which will be compiled and run on the IPU. This is done using the `ipu.loops` API. The functions in the API take a Python function and some inputs, apply the function repeatedly to the inputs, and return the output of the final application of the function.

The body of the loop can get its inputs from two different places: its previous output or the next element of an infeed queue. It can take inputs from both simultaneously. If a loop body takes inputs from both, the inputs from the previous output must appear first in the list of arguments. Inputs from an infeed queue are unpacked into the loop body arguments. Examples of cases where taking the previous output as input is useful include getting summary statistics (such as our running total of the loss) and any model where the same layer or layers are applied repeatedly.

These points are illustrated by our new loop body:

```python
# SNIPPET 3

# "Previous output" arguments come before infeed queue arguments
def training_loop_body(loss_running_total, x, y):
    logits = model(x, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss=loss)

    # The running total calculation is now moved to the IPU
    return([loss_running_total + loss, train_op])
```



Now we can create the loop itself. There are two functions for forming loops available: 
- `ipu.loops.repeat`, which executes the loop body a fixed number of times
- `ipu.loops.while`, which executes the loop body until some condition is met

Both take the following arguments:

- `body`, a Python function which gives the body of the loop
- `inputs`, a list of initial values for the "previous output" inputs
- `infeed_queue`, an infeed queue from which to take inputs

For both types of loop, `inputs` and `infeed_queue` are optional arguments (that is, there is no requirement for a loop to carry variables or take data from an infeed queue). 

Each has one more important argument, specific to the kind of loop they execute:

- `ipu.loops.repeat`: `n`, the number of times to execute the loop
- `ipu.loops.while`: `condition`, a callable returning a Boolean scalar tensor that determines if execution continues, as in `tf.while_loop`

Here's how that works out for our example, with comments briefly explaining each argument:

```python
# SNIPPET 4

def train_one_epoch():

    total_loss = ipu.loops.repeat(

        # Repeat same number of times as before
        n=batches_per_epoch,

        # The training loop body we defined in the previous snippet
        body=training_loop_body,

        # Set initial value of running total to 0
        inputs=[0.0],

        # We use our infeed queue from the previous section
        infeed_queue=mnist_infeed_queue)

    return total_loss
```  

## Configure, prepare and run

Configuring the IPU and preparing the model are the same as before, except that since the data is now fed from an infeed queue, we do not need to specify the inputs when we use `ipu.ipu_compiler.compile`.


```python
# SNIPPET 5

ipu_configuration = ipu.utils.create_ipu_config()

ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)

ipu.utils.configure_ipu_system(config=ipu_configuration)

with ipu.scopes.ipu_scope('/device:IPU:0'):
    train_one_epoch_on_ipu = ipu.ipu_compiler.compile(train_one_epoch)
```

The only change that needs to be made to the code apart from removing the inner loop is to initialise the infeed queue in place of initialising the dataset iterator as we did in Part 1. 

Here's the final part of the example:

```python
# SNIPPET 6

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Initialize the infeed queue
    sess.run(mnist_infeed_queue.initializer)

    for i in range(EPOCHS):

        epoch_start_time = time.time()

        # Inner loop
        total_loss = sess.run(train_one_epoch_on_ipu)

        # Print average loss and time taken for epoch
        print('\n', end='')
        print("Loss:", total_loss[0]/batches_per_epoch)
        print("Time:", time.time() - epoch_start_time)

print("Program ran successfully")
```

And we're done! You should see that this change results in a dramatic increase in throughput compared to the code from Part 1.

We can examine the reasons for the speedup using the PopVision System Analyser, which is available for download [here](https://downloads.graphcore.ai/) and is documented [here](https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/system/system.html). When the `ipu.loops` API is not used, the program that executes the training loop has to be destroyed and rebuilt between batches. When it is used, there is no need and the program can run continuously.

![PopVision System Analyser trace without `ipu.loops`](system_trace_without_ipu_loops.png)
![PopVision System Analyser trace with `ipu.loops`](system_trace_with_ipu_loops.png)


