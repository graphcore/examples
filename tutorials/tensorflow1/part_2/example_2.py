# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


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


# SNIPPET 2

mnist_infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="mnist_data")


# SNIPPET 3

# "Previous output" arguments come before infeed queue arguments
def training_loop_body(loss_running_total, x, y):
    logits = model(x, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss=loss)

    # The running total calculation is now moved to the IPU
    return([loss_running_total + loss, train_op])


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


# SNIPPET 5

ipu_configuration = ipu.utils.create_ipu_config()

ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)

ipu.utils.configure_ipu_system(config=ipu_configuration)

with ipu.scopes.ipu_scope('/device:IPU:0'):
    train_one_epoch_on_ipu = ipu.ipu_compiler.compile(train_one_epoch)


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
