# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Example of training a simple convolutional model on Fashion-MNIST using mixed precision

import time
import argparse

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

# Process command-line arguments

parser = argparse.ArgumentParser(
    description='Train a simple convolutional model on Fashion-MNIST')

parser.add_argument('chosen_precision_str',
                    metavar='precision',
                    choices=['mixed', 'float32'],
                    type=str,
                    help='Precision to use')

parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='Batch size to use for training')

parser.add_argument('--epochs',
                    type=int,
                    default=5,
                    help='Number of epochs to train for')

parser.add_argument('--loss-scaling-factor',
                    type=float,
                    default=2**8,
                    help='Scaling factor for loss scaling')

parser.add_argument('--learning-rate',
                    type=float,
                    default=0.01,
                    help='Learning rate for the optimizer')

parser.add_argument('--use-float16-partials',
                    action='store_true',
                    help='Use FP16 partials in matmuls and convs')

args = parser.parse_args()

# Computations done in chosen precision, which is float16 for mixed precision
if args.chosen_precision_str == 'mixed':
    compute_precision_str, compute_precision = 'float16', tf.float16
else:
    compute_precision_str, compute_precision = 'float32', tf.float32

# Load the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), _ = fashion_mnist.load_data()

# Normalize and cast the data
x_train = x_train.astype(compute_precision_str) / 255
y_train = y_train.astype('int32')

# Create and configure the dataset for iteration
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().batch(args.batch_size, drop_remainder=True)

# Create an infeed queue
mnist_infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="mnist_data")

# Use floor division because we drop the remainder
batches_per_epoch = len(x_train) // args.batch_size


# FP32 parameter getter
# This function creates FP32 weights no matter what the compute dtype is

def fp32_parameter_getter(getter, name, dtype, trainable, shape=None, *args, **kwargs):

    if trainable and dtype != tf.float32:
        parameter_variable = getter(name, shape, tf.float32, *args, trainable=trainable, **kwargs)
        return tf.cast(parameter_variable, dtype=dtype, name=name + "_cast")

    else:
        parameter_variable = getter(name, shape, dtype, *args, trainable=trainable, **kwargs)
        return parameter_variable


# Define a convolution that uses tf.get_variable to create the kernel
# We use different `op_name`s for each operation so the variables are all given different names
def conv(feature_map, kernel_size, stride, filters_out, op_name, padding='SAME'):

    # We use NHWC format
    filters_in = feature_map.get_shape().as_list()[-1]

    # Resource variables must be used on the IPU
    with tf.variable_scope(op_name, use_resource=True):

        kernel = tf.get_variable(
            name="conv2d/kernel",
            shape=[kernel_size, kernel_size, filters_in, filters_out],
            dtype=feature_map.dtype,
            trainable=True
        )

        return tf.nn.conv2d(
            feature_map,
            filters=kernel,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format="NHWC",
        )


# Define a dense layer that uses tf.get_variable to create the weights and biases
def dense(inputs, units_out, op_name):

    flattened_inputs = tf.layers.flatten(inputs)

    flattened_inputs_size = flattened_inputs.get_shape().as_list()[-1]

    # Expand dimensions to do batched matmul
    flattened_inputs = tf.expand_dims(flattened_inputs, -1)

    with tf.variable_scope(op_name, use_resource=True):

        weights = tf.get_variable(
            name="weights",
            shape=[units_out, flattened_inputs_size],
            dtype=inputs.dtype,
            trainable=True
        )

        biases = tf.get_variable(
            name="biases",
            shape=[units_out, 1],
            dtype=inputs.dtype,
            trainable=True
        )

        return tf.matmul(weights, flattened_inputs) + biases


# Define a function that applies the model function to some inputs
# This model is contrived, but is not the focus of this tutorial
def model_function(input_image_batch):

    layer_out = tf.reshape(input_image_batch, [args.batch_size, 28, 28, 1])

    layer_out = conv(layer_out, kernel_size=3, stride=1, filters_out=32, op_name='conv1')

    layer_out = tf.nn.relu(layer_out)

    layer_out = conv(layer_out, kernel_size=3, stride=1, filters_out=32, op_name='conv2')

    layer_out = dense(layer_out, units_out=10, op_name='dense')

    return layer_out


# Define the body of the training loop, to pass to `ipu.loops.repeat`
def training_loop_body(loss_running_total, x, y):

    # Apply the model function to the inputs
    # Using the chosen variable getter as our custom getter
    with tf.variable_scope('all_vars', use_resource=True,
                           custom_getter=fp32_parameter_getter):
        logits = model_function(x)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    # When using Adam in FP16, you should check
    #     the default value of epsilon and ensure
    #     that it does not underflow
    optimizer = tf.train.AdamOptimizer(args.learning_rate, epsilon=1e-4)

    # Scale loss
    loss *= args.loss_scaling_factor

    # Calculate gradients with scaled loss
    grads_and_vars = optimizer.compute_gradients(loss=loss)

    # Rescale gradients to correct values so parameter update step is correct
    grads_and_vars = [(gradient/args.loss_scaling_factor, variable)
                      for gradient, variable in grads_and_vars]

    # Apply gradients
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    # Return loss to original value before reporting it
    loss /= args.loss_scaling_factor

    return([loss_running_total + loss, train_op])


# Use `ipu.loops.repeat` to train for one epoch
def train_one_epoch():

    total_loss = ipu.loops.repeat(
        n=batches_per_epoch,
        body=training_loop_body,
        # Set initial value of loss running total to 0
        inputs=tf.constant([0.0], dtype=compute_precision),
        infeed_queue=mnist_infeed_queue
    )

    return total_loss

# Configure device with 1 IPU and compile

ipu_configuration = ipu.utils.create_ipu_config()

ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration,
                                               num_ipus=1)

# Explicitly enable all floating-point exceptions and disable stochastic rounding
ipu_configuration = ipu.utils.set_floating_point_behaviour_options(opts=ipu_configuration,
                                                                   esr=False,
                                                                   nanoo=True,
                                                                   oflo=True, inv=True, div0=True)

if args.use_float16_partials:

    ipu_configuration = ipu.utils.set_matmul_options(
        opts=ipu_configuration,
        matmul_options={'partialsType': 'half'})

    ipu_configuration = ipu.utils.set_convolution_options(
        opts=ipu_configuration,
        convolution_options={'partialsType': 'half'})

ipu.utils.configure_ipu_system(config=ipu_configuration)

with ipu.scopes.ipu_scope('/device:IPU:0'):
    train_one_epoch_on_ipu = ipu.ipu_compiler.compile(train_one_epoch)

# Run training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Initialize the infeed queue
    sess.run(mnist_infeed_queue.initializer)

    # Run one epoch before profiling so we can ignore compilation time
    print("Executing warmup run...")

    _ = sess.run(train_one_epoch_on_ipu)

    # Train the model, timing how long each epoch takes
    t0 = time.time()
    for i in range(args.epochs):

        total_loss = sess.run(train_one_epoch_on_ipu)
        print("Average loss:", total_loss[0] / batches_per_epoch, "--- time so far", time.time()-t0)

    # Report throughput
    throughput = args.batch_size * batches_per_epoch * args.epochs / (time.time() - t0)
    print("throughput", throughput, "images/s")

print("Program ran successfully")
