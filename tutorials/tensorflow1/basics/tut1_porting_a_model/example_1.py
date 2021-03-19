# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


# SNIPPET 1

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

# We'll be interested in performance later
import time


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


# SNIPPET 3

# Create a default configuration
ipu_configuration = ipu.utils.create_ipu_config()

# Select an IPU automatically
ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)

# Apply the configuration
ipu.utils.configure_ipu_system(config=ipu_configuration)


# SNIPPET 4

def training_loop_body(x, y):

    logits = model(x, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)

    return([loss, train_op])


# SNIPPET 5

# Get inputs from get_next() method of iterator
(x, y) = dataset_iterator.get_next()

# We build the operation within the scope of a particular device
with ipu.scopes.ipu_scope('/device:IPU:0'):

    # Pass the training loop function and list of inputs to ipu.ipu_compiler.compile
    training_loop_body_on_ipu = ipu.ipu_compiler.compile(computation=training_loop_body, inputs=[x, y])


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
