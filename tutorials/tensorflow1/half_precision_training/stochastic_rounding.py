# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Example of training ResNet on CIFAR-10 using stochastic rounding

import sys
import time
import argparse

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

# Process command-line arguments

parser = argparse.ArgumentParser(
    description='Train ResNet on CIFAR-10 using stochastic rounding')

parser.add_argument('chosen_precision_str',
                    metavar='precision',
                    choices=['float16', 'float32'],
                    type=str,
                    help='Precision in which to perform computations')

parser.add_argument('number_of_layers',
                    type=int,
                    help='Number of layers in ResNet to use')

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

chosen_precision = tf.float32 if args.chosen_precision_str == 'float32' else tf.float16

# Check the given number of layers defines a valid ResNet
if args.number_of_layers % 6 != 2 or args.number_of_layers < 8:

    print('Please choose a number of layers of the form '
          '6N + 2 for a whole number N, such as 8, 14, 20, 26, etc. '
          '(Note that valid sizes for CIFAR images differ from those '
          'for ImageNet images)')

    sys.exit(1)

# Load the data
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), _ = cifar10.load_data()

# Normalize and cast the data
x_train = x_train.astype(args.chosen_precision_str) / 255
y_train = y_train.astype('int32')

# Create and configure the dataset for iteration
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().batch(args.batch_size, drop_remainder=True)

# Create an infeed queue
cifar10_infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="cifar10_data")

# Use floor division because we drop the remainder
batches_per_epoch = len(x_train) // args.batch_size

# Define the residual blocks for use in our model

# We use ordinary Keras layers here only to define the model function


# Ordinary residual block
# We use the identity function for the skip connection
def residual_block(inputs, filters):

    conv1_layer = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         activation=tf.nn.relu,
                                         data_format="channels_last")

    conv1_output = conv1_layer(inputs)

    conv2_layer = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         activation=tf.nn.relu,
                                         data_format="channels_last")

    conv2_output = conv2_layer(conv1_output)

    sum_of_paths = tf.keras.layers.add([conv2_output, inputs])

    output = tf.keras.layers.Activation('relu')(sum_of_paths)

    return output


# Residual block with a downsample convolution
# We use a 1x1 conv with stride 2 for the skip connection
def downsample_residual_block(inputs, filters):

    conv1_layer = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 3),
                                         strides=(2, 2),
                                         padding='same',
                                         activation=tf.nn.relu,
                                         data_format="channels_last")

    conv1_output = conv1_layer(inputs)

    conv2_layer = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         activation=tf.nn.relu,
                                         data_format="channels_last")

    conv2_output = conv2_layer(conv1_output)

    skip_connection_conv = tf.keras.layers.Conv2D(filters=filters,
                                                  kernel_size=(1, 1),
                                                  strides=(2, 2),
                                                  padding='same',
                                                  activation=tf.nn.relu,
                                                  data_format="channels_last")

    skip_connection_output = skip_connection_conv(inputs)

    sum_of_paths = tf.keras.layers.add([conv2_output, skip_connection_output])

    output = tf.keras.layers.Activation('relu')(sum_of_paths)

    return output

# Create the model using the Keras functional API
# We use batch normalisation between blocks

inputs = tf.keras.Input(shape=(32, 32, 3), dtype=chosen_precision)

# First layer is 3x3 convolution with 16 filters
first_conv_layer = tf.keras.layers.Conv2D(filters=16,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          activation=tf.nn.relu,
                                          data_format="channels_last")

block_output = first_conv_layer(inputs)

blocks_per_section = (args.number_of_layers - 2) // 6

# Apply first set of residual blocks
for _ in range(blocks_per_section):

    block_output = tf.keras.layers.BatchNormalization()(block_output)
    block_output = residual_block(block_output, filters=16)

# Downsample from 32x32 with 16 filters to 16x16 with 32 filters
block_output = tf.keras.layers.BatchNormalization()(block_output)
block_output = downsample_residual_block(block_output, filters=32)

# Apply second set of residual blocks
for _ in range(blocks_per_section - 1):

    block_output = tf.keras.layers.BatchNormalization()(block_output)
    block_output = residual_block(block_output, filters=32)

# Downsample from 16x16 with 32 filters to 8x8 with 64 filters
block_output = tf.keras.layers.BatchNormalization()(block_output)
block_output = downsample_residual_block(block_output, filters=64)

# Apply third set of residual blocks
for _ in range(blocks_per_section - 1):

    block_output = tf.keras.layers.BatchNormalization()(block_output)
    block_output = residual_block(block_output, filters=64)

# Finish with global average pool and dense layer
global_average_pool = tf.keras.layers.GlobalAveragePooling2D()(block_output)

reshape = tf.keras.layers.Reshape((64,))(global_average_pool)

outputs = tf.keras.layers.Dense(10)(reshape)

# With all ops defined, create the model from the inputs and outputs
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# Define the body of the training loop, to pass to `ipu.loops.repeat`
def training_loop_body(loss_running_total, x, y):

    logits = model(x, training=True)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    # Apply loss scaling
    loss *= args.loss_scaling_factor

    # Adjust learning rate so parameter update step is correct
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=args.learning_rate/args.loss_scaling_factor,
        momentum=0.9)

    train_op = optimizer.minimize(loss=loss)

    # Return loss to original value before reporting it
    loss /= args.loss_scaling_factor

    return([loss_running_total + loss, train_op])


# Use `ipu.loops.repeat` to train for one epoch
def train_one_epoch():

    total_loss = ipu.loops.repeat(
        n=batches_per_epoch,
        body=training_loop_body,
        # Set initial value of loss running total to 0
        inputs=[tf.constant(0.0, dtype=chosen_precision)],
        infeed_queue=cifar10_infeed_queue)

    return total_loss

# Configure device with 1 IPU and compile

ipu_configuration = ipu.utils.create_ipu_config()

ipu_configuration = ipu.utils.auto_select_ipus(opts=ipu_configuration, num_ipus=1)

# Enable stochastic rounding
# We also explicitly enable all floating-point exceptions
ipu_configuration = ipu.utils.set_floating_point_behaviour_options(opts=ipu_configuration,
                                                                   esr=True,
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
    sess.run(cifar10_infeed_queue.initializer)

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
