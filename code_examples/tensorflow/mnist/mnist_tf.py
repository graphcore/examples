# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from tqdm import tqdm
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import utils, ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu import dataset_benchmark
from tensorflow.python.ipu import rand_ops
import argparse
tf.disable_eager_execution()
tf.disable_v2_behavior()


def dense_layer(hiddenSize, input, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE, use_resource=True):
        w = tf.get_variable("weight", shape=[input.shape[-1], hiddenSize],
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("bias", shape=[hiddenSize],
                            initializer=tf.zeros_initializer())
        return tf.nn.relu_layer(input, w, b)


def model(lr, outqueue, training: bool, inputs, labels):
    h1Size = 320
    h2Size = 10
    droprate = 0.2

    relu1 = dense_layer(h1Size, inputs, "d1")
    # Use the IPU optimised version of dropout:
    if training:
        drop1 = rand_ops.dropout(relu1, rate=droprate)
    else:
        drop1 = relu1
    relu2 = dense_layer(h2Size, drop1, "d2")

    with tf.variable_scope("metrics", reuse=tf.AUTO_REUSE, use_resource=True):
        acc, acc_op = tf.metrics.accuracy(labels=labels,
                                          predictions=tf.argmax(
                                              relu2, axis=1, output_type=tf.dtypes.int32),
                                          name="accuracy")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=relu2)

    if training:
        with tf.variable_scope("training", reuse=tf.AUTO_REUSE, use_resource=True):
            optimiser = tf.train.MomentumOptimizer(
                learning_rate=lr, momentum=0.0001, use_nesterov=True, name='optimise')
            train_op = optimiser.minimize(loss)
            with tf.control_dependencies([train_op, acc_op]):
                mean_loss = tf.reduce_mean(loss, name='train_loss')
    else:
        with tf.control_dependencies([acc_op]):
            mean_loss = tf.reduce_mean(loss, name='test_loss')

    return outqueue.enqueue({'mean_loss': mean_loss, 'acc': acc})


def scheduler(epoch):
    if epoch < 1:
        return 0.02
    if epoch < 3:
        return 0.01
    else:
        return 0.001


def loop_builder(iterations, builder_func, infeed):
    return loops.repeat(iterations, builder_func, [], infeed)


def run_model(opts):
    # Use Keras to get the dataset:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Sizes/shapes for the dataset:
    image_shape = x_train.shape[1:]
    num_pixels = image_shape[0] * image_shape[1]
    batch_size = 16
    batch_shape = [batch_size, num_pixels]
    num_train = y_train.shape[0]
    num_test = y_test.shape[0]
    data_shape = [None, num_pixels]

    # Flatten the images and cast the labels:
    x_train_flat = x_train.astype(np.float32).reshape(-1, num_pixels)
    x_test_flat = x_test.astype(np.float32).reshape(-1, num_pixels)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Decide how to split epochs into loops up front:
    epochs = 5
    ipu_steps_per_epoch = 15
    batches_per_epoch = num_train // batch_size
    train_batches = (num_train * epochs) // batch_size
    test_batches = num_test // batch_size
    batches_per_step = batches_per_epoch // ipu_steps_per_epoch
    if not batches_per_epoch % ipu_steps_per_epoch == 0:
        raise ValueError(f"IPU steps per epoch {ipu_steps_per_epoch} must divide batches per epoch {batches_per_epoch}.")

    # Put placeholders on the CPU host:
    with tf.device("cpu"):
        place_x = tf.placeholder(dtype=tf.float32, shape=data_shape, name="input")
        place_y = tf.placeholder(dtype=tf.int32, shape=[None], name="label")
        lr_placeholder = tf.placeholder(tf.float32, shape=[])

    # Create dataset and IPU feeds:
    dataset = tf.data.Dataset.from_tensor_slices((place_x, place_y))
    dataset = dataset.cache().repeat().batch(batch_size, drop_remainder=True)
    infeed_train_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, feed_name="train_infeed")
    outfeed_train_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name="train_outfeed")
    infeed_test_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, feed_name="test_infeed")
    outfeed_test_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name="test_outfeed")

    # Use function binding to create all the builder functions that are neeeded:
    bound_train_model = partial(model, lr_placeholder, outfeed_train_queue, True)
    bound_train_loop = partial(
        loop_builder, batches_per_step, bound_train_model, infeed_train_queue)
    bound_test_model = partial(model, lr_placeholder, outfeed_test_queue, False)
    bound_test_loop = partial(loop_builder, test_batches,
                              bound_test_model, infeed_test_queue)

    # Use the bound builder functions to place the model on the IPU:
    with scopes.ipu_scope("/device:IPU:0"):
        train_loop = ipu_compiler.compile(bound_train_loop, inputs=[])
        test_loop = ipu_compiler.compile(bound_test_loop, inputs=[])

    # Initialisers should go on the CPU:
    with tf.device("cpu"):
        metrics_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        metrics_initializer = tf.variables_initializer(var_list=metrics_vars)
        saver = tf.train.Saver()

    # Setup and acquire an IPU device:
    config = utils.create_ipu_config()
    config = utils.auto_select_ipus(config, 1)
    utils.configure_ipu_system(config)

    # These allow us to retrieve the results of IPU feeds:
    dequeue_train_outfeed = outfeed_train_queue.dequeue()
    dequeue_test_outfeed = outfeed_test_queue.dequeue()

    # Create a benchmark program for the infeed to determine maximum achievable throughput:
    infeed_perf = dataset_benchmark.infeed_benchmark(
        infeed_train_queue, epochs, num_train, True)

    print(f"\nImage shape: {image_shape} Training examples: {num_train} Test examples: {num_test}")
    print(f"Epochs: {epochs} Batch-size: {batch_size} Steps-per-epoch: {ipu_steps_per_epoch} Batches-per-step: {batches_per_step}")

    # Run the model:
    with tf.Session() as sess:
        print(f"Benchmarking the infeed...")
        sess.run(infeed_perf, feed_dict={place_x: x_train_flat, place_y: y_train})

        sess.run(tf.global_variables_initializer())
        sess.run(infeed_train_queue.initializer, feed_dict={
                 place_x: x_train_flat, place_y: y_train})

        if opts.test_mode in ["all", "training"]:
            print(f"Training...")
            progress = tqdm(
                range(epochs), bar_format='{desc} Epoch: {n_fmt}/{total_fmt} {bar}')
            for e in progress:

                sess.run(metrics_initializer)
                for i in range(ipu_steps_per_epoch):
                    sess.run(train_loop, feed_dict={lr_placeholder: scheduler(e)})
                    result = sess.run(dequeue_train_outfeed)
                    if len(result['mean_loss'] != 0) and len(result['acc'] != 0):
                        progress.set_description(f"Loss {result['mean_loss'][0]:.5f} Accuracy {result['acc'][0]:.5f}")

            print(f"Saving...")
            saver.save(sess, "model")

        if opts.test_mode in ["all", "tests"]:
            print(f"Testing...")
            sess.run(metrics_initializer)
            sess.run(infeed_test_queue.initializer, feed_dict={
                     place_x: x_test_flat, place_y: y_test})
            sess.run(test_loop)
            result = sess.run(dequeue_test_outfeed)

            test_loss = np.mean(result['mean_loss'])
            test_acc = np.mean(result['acc'])
            print(f"Test loss: {test_loss:.8f} Test accuracy: {test_acc:.8f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and Test the simple Tensorflow model with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--test-mode",
        choices=['training', 'tests', 'all'],
        default="all",
        help="Use this flag to run the model in either 'training' or 'tests' mode or both ('all')")

    opts = parser.parse_args()
    run_model(opts)
