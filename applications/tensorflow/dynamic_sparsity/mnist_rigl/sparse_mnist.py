# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from tqdm import tqdm
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import utils, ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu import rand_ops
import argparse
import json
import time
import os
from datetime import datetime
from ipu_sparse_ops import sparse, layers, optimizers, sparse_training
import logging

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = logging.getLogger(os.path.basename(__file__))


def build_optimizer(opt_name, opt_args=None):
    # Fetch the requested optimiser
    opt_cls = {
        'GradientDescent': tf.train.GradientDescentOptimizer,
        'Momentum': tf.train.MomentumOptimizer,
        'Adam': tf.train.AdamOptimizer
    }.get(opt_name)

    if opt_cls is None:
        raise ValueError(f'Unsupported optimizer {opt_name}')

    # Fetch default kwargs, accepting overrides from argparse
    opt_kws = {
        'GradientDescent': {},
        'Momentum': {
            'momentum': 0.0001,
            'use_nesterov': True
        },
        'Adam': {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-08
        }
    }.get(opt_name)
    if opt_args is not None:
        opt_kws.update(opt_args)

    return opt_cls, opt_kws


def model(fc_layers, droprate, lr, opt_cls, opt_kws, iterations_per_step, training: bool,
          last_outqueue, inputs, labels):

    with tf.variable_scope("counter", reuse=tf.AUTO_REUSE, use_resource=True):
        itr_counter = tf.get_variable("iterations", shape=[], dtype=tf.int32,
                                      trainable=False,
                                      initializer=tf.zeros_initializer())
        mod_itrs = tf.math.floormod(itr_counter, iterations_per_step)
        last_itr = tf.equal(mod_itrs, 0)
        inc = tf.assign_add(itr_counter, 1)

    fc1 = fc_layers['fc1']
    fc2 = fc_layers['fc2']

    relu1 = fc1(inputs, last_itr)

    # Use the IPU optimised version of dropout:
    if training:
        drop1 = rand_ops.dropout(relu1, rate=droprate)
    else:
        drop1 = relu1

    relu2 = fc2(drop1, last_itr)

    with tf.variable_scope("metrics", reuse=tf.AUTO_REUSE, use_resource=True):
        acc, acc_op = tf.metrics.accuracy(labels=labels,
                                          predictions=tf.argmax(
                                              relu2, axis=1, output_type=tf.dtypes.int32),
                                          name="accuracy")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=tf.cast(relu2, dtype=tf.float32, name="logits_to_fp32"))

    if training:
        with tf.variable_scope("training", reuse=tf.AUTO_REUSE, use_resource=True):
            optimiser = optimizers.SparseOptimizer(opt_cls)(
                learning_rate=lr, **opt_kws, name='optimise',
                sparse_layers=[fc1, fc2])
            train_op = optimiser.minimize(loss)
            slot_names = optimiser.get_slot_names()
            logger.debug(f"Optimiser slot names: {slot_names}")
            with tf.control_dependencies([train_op, acc_op]):
                mean_loss = tf.reduce_mean(loss, name='train_loss')
    else:
        with tf.control_dependencies([acc_op]):
            mean_loss = tf.reduce_mean(loss, name='test_loss')

    # Prepare results for feeds:
    last_results = {'mean_loss': mean_loss, 'acc': acc}
    for name, fc in fc_layers.items():
        if fc.is_sparse():
            weights_tensor = tf.convert_to_tensor(fc.get_values_var())
            last_results[name + '_non_zeros'] = weights_tensor
            if training:
                dense_grad_w = fc.get_dense_grad_w(loss)
                last_results[name + '_grad_w'] = tf.convert_to_tensor(dense_grad_w)

                for slot_name in fc.sparse_slots:
                    last_results[name + f'_{slot_name}'] = \
                        tf.convert_to_tensor(fc.sparse_slots[slot_name].tf_variable)

    # When training we only want to return the sparse
    # non-zero weight values on the last iteration.
    if training:
        def enqueue_last_itr():
            enqueue_weights = last_outqueue.enqueue(last_results)
            with tf.control_dependencies([enqueue_weights]):
                return tf.no_op()

        def nop():
            return tf.no_op()

        cond_op = tf.cond(last_itr, enqueue_last_itr, nop)
        enqueue_op = tf.group(inc, cond_op, train_op)
    else:
        enqueue_op = last_outqueue.enqueue({'mean_loss': mean_loss, 'acc': acc})

    return enqueue_op


def create_fc_layers(opts, batch_shape, random_gen):
    h1 = 320
    h2 = 10
    batch_size = batch_shape[0]
    density1, density2 = opts.densities
    make_sparse = layers.SparseFcLayer.from_random_generator
    dtype = tf.float16 if opts.data_type == 'fp16' else tf.float32

    fc_layers = {}
    if density1 >= 1:
        fc_layers['fc1'] = layers.DenseFcLayer(h1, name='dense_fc',
                                               dtype=dtype, bias=True, relu=True)
    else:
        limit = np.sqrt(6/((batch_shape[1] + h1)*density1))
        glorot_uniform_gen = partial(random_gen.uniform, -limit, limit)
        indices_random_gen = np.random.default_rng(seed=opts.seed)
        options = {"metaInfoBucketOversizeProportion": 0.2}
        fc_layers['fc1'] = make_sparse(h1, batch_shape, density1,
                                       glorot_uniform_gen,
                                       indices_random_gen,
                                       matmul_options=options,
                                       name='sparse_fc',
                                       dtype=dtype,
                                       bias=True, relu=True)
    if density2 >= 1:
        fc_layers['fc2'] = layers.DenseFcLayer(h2, name='dense_classifier',
                                               dtype=dtype, bias=True, relu=False)
    else:
        limit = np.sqrt(6/((h1 + h2)*density2))
        glorot_uniform_gen = partial(random_gen.uniform, -limit, limit)
        indices_random_gen = np.random.default_rng(seed=opts.seed)
        options = {"metaInfoBucketOversizeProportion": 0.1}
        fc_layers['fc2'] = make_sparse(h2, [batch_size, h1], density2,
                                       glorot_uniform_gen,
                                       indices_random_gen,
                                       matmul_options=options,
                                       name='sparse_classifier',
                                       dtype=dtype,
                                       bias=True, relu=False)
    return fc_layers


def build_update_op(fc_layers):
    # Need to build update ops for each sparse layer so that
    # we can change sparsity pattern during training:
    fc_update_ops = {}
    for name, fc in fc_layers.items():
        if fc.is_sparse():
            fc_update_ops[name] = fc.update_sparsity_op()
            if not opts.disable_pruning:
                # If a layer's sparsity pattern changed then its slot
                # also need to be updated:
                fc_update_ops[name + '_slots'] = fc.update_slots_op()

    # Combine all layer updates into one update op:
    return tf.group(fc_update_ops.values())


def prune_and_grow(layer_name, layer, outputs_from_last_step, random_gen,
                   step, total_steps, opts):

    def cosine_prune_schedule(t, T, max_pruned):
        return int(np.ceil(.5 * max_pruned * (1 + np.cos(t * (np.pi/T)))))

    name = layer_name
    fc = layer
    last = outputs_from_last_step

    # Sync the layer's internal host-side state with last results
    # (both weights and slots need to be kept in sync):
    fc.sync_internal_representation(
        last[name+'_non_zeros'][0],
        {
            slot_name: last[name+f'_{slot_name}'][0]
            for slot_name in fc.sparse_slots
        })

    updater = partial(sparse_training.prune_and_grow,
                      prune_schedule=partial(cosine_prune_schedule, t=step, T=total_steps),
                      prune_ratio=opts.prune_ratio,
                      grad_w=np.array(last[name+'_grad_w'][0]),
                      grow_method=opts.regrow,
                      random_gen=np.random.default_rng(seed=opts.seed))

    fc.update_sparsity_pattern(updater)

    if opts.records_path and name == 'fc1':
        # Save the first hidden layer's weight mask for later analysis:
        save_weights(opts, name, fc, step)


def save_weights(opts, name, fc, step):
    if fc.is_sparse():
        os.makedirs(opts.records_path, exist_ok=True)
        filename = os.path.join(opts.records_path, f"weights_{name}_{step:06}")
        np.save(filename, fc.extract_dense())


def scheduler(epoch, opts):
    progress = epoch / opts.epochs
    lr_scale = 16/opts.batch_size

    if opts.optimizer == "Adam":
        return 0.002 * lr_scale

    if progress < 0.2:
        return 0.02 * lr_scale
    if progress < 0.4:
        return 0.01 * lr_scale
    if progress < 0.6:
        return 0.005 * lr_scale
    if progress < 0.8:
        return 0.001 * lr_scale
    else:
        return 0.0001 * lr_scale


def loop_builder(iterations, builder_func, infeed):
    return loops.repeat(iterations, builder_func, [], infeed)


def run_mnist(opts):
    if opts.seed is not None:
        utils.reset_ipu_seed(opts.seed)
    random_gen = np.random.default_rng(seed=opts.seed)

    # Use Keras to get the dataset:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Sizes/shapes for the dataset:
    image_shape = x_train.shape[1:]
    num_pixels = image_shape[0] * image_shape[1]
    batch_size = opts.batch_size
    batch_shape = [batch_size, num_pixels]
    num_train = y_train.shape[0]
    num_test = y_test.shape[0]
    data_shape = [None, num_pixels]
    dtype = tf.float16 if opts.data_type == 'fp16' else tf.float32

    # Flatten the images and cast the labels:
    x_train_flat = x_train.astype(dtype.as_numpy_dtype()).reshape(-1, num_pixels)
    x_test_flat = x_test.astype(dtype.as_numpy_dtype()).reshape(-1, num_pixels)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Decide how to split epochs into loops up front:
    batches_per_epoch = num_train // batch_size
    train_batches = (num_train * opts.epochs) // batch_size
    test_batches = num_test // batch_size
    batches_per_step = batches_per_epoch // opts.steps_per_epoch
    if not batches_per_epoch % opts.steps_per_epoch == 0:
        raise ValueError(f"IPU steps per epoch {opts.steps_per_epoch} must divide batches per epoch {batches_per_epoch} exactly.")

    # Create FC layer descriptions:
    fc_layers = create_fc_layers(opts, batch_shape, random_gen)
    for name, fc in fc_layers.items():
        logger.info(f"Layer Config: {name}: {type(fc)}")

    # Put placeholders on the CPU host:
    with tf.device("cpu"):
        place_x = tf.placeholder(dtype=dtype, shape=data_shape, name="input")
        place_y = tf.placeholder(dtype=tf.int32, shape=[None], name="label")
        lr_placeholder = tf.placeholder(dtype, shape=[])

    # Create dataset and IPU feeds:
    dataset = tf.data.Dataset.from_tensor_slices((place_x, place_y))
    dataset = dataset.shuffle(buffer_size=num_train, seed=opts.seed).cache()
    dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
    infeed_train_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, feed_name="train_infeed")
    outfeed_train_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name="train_outfeed_last_itr")
    infeed_test_queue = ipu_infeed_queue.IPUInfeedQueue(
        dataset, feed_name="test_infeed")
    outfeed_test_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        feed_name="test_outfeed")

    # Get optimiser
    opt_cls, opt_kws = build_optimizer(opts.optimizer, opts.optimizer_arg)
    logger.info('Optimiser %s, optimiser keywords %s', opt_cls.__name__, opt_kws)

    # Use function binding to create all the builder functions that are needed:
    bound_train_model = partial(
        model, fc_layers, opts.droprate, lr_placeholder,
        opt_cls, opt_kws, batches_per_step,
        True, outfeed_train_queue)
    bound_train_loop = partial(
        loop_builder, batches_per_step, bound_train_model, infeed_train_queue)
    bound_test_model = partial(
        model, fc_layers, opts.droprate, lr_placeholder,
        opt_cls, opt_kws, batches_per_step,
        False, outfeed_test_queue)
    bound_test_loop = partial(
        loop_builder, test_batches,
        bound_test_model, infeed_test_queue)

    # Use the bound builder functions to place the model on the IPU:
    with scopes.ipu_scope("/device:IPU:0"):
        train_loop = ipu_compiler.compile(bound_train_loop, inputs=[])
        test_loop = ipu_compiler.compile(bound_test_loop, inputs=[])

    # Placeholders can only be created on cpu after all the slots have registered:
    with tf.device("cpu"):
        for fc in fc_layers.values():
            fc.create_placeholders()

    # Create update op on IPU:
    with scopes.ipu_scope("/device:IPU:0"):
        update_representation = build_update_op(fc_layers)

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
    dequeue_test_outfeed = outfeed_test_queue.dequeue()
    dequeue_train_outfeed = outfeed_train_queue.dequeue()

    logger.info(f"Image shape: {image_shape} Training examples: {num_train} Test examples: {num_test}")
    logger.info(f"Epochs: {opts.epochs} Batch-size: {batch_size} Steps-per-epoch: {opts.steps_per_epoch} Batches-per-step: {batches_per_step}")
    total_steps = opts.steps_per_epoch * opts.epochs
    logger.info(f"Total steps: {total_steps}")

    if opts.log:
        # Open log and write header fields:
        log_file = open(opts.log, 'w')
        d1, d2 = opts.densities
        log_file.write(f"Iteration Density_{d1}_{d2}\n")

    logpath = os.path.join(opts.checkpoint_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.FileWriter(logpath)

    if opts.records_path:
        # Save the first hidden layer's weight mask for later analysis:
        save_weights(opts, 'fc1', fc_layers['fc1'], 0)

    # Run the model:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(infeed_train_queue.initializer, feed_dict={
                 place_x: x_train_flat, place_y: y_train})

        if opts.test_mode in ["all", "training"]:
            logger.info(f"Training...")
            progress = tqdm(
                range(opts.epochs), bar_format='{desc} Epoch: {n_fmt}/{total_fmt} {bar}')
            for e in progress:
                for i in range(opts.steps_per_epoch):
                    sess.run(metrics_initializer)
                    # Only need to feed an updated sparsity representation if we are running
                    # a prune and grow algorithm:
                    if not opts.disable_pruning:
                        # Merge the feeds needed for all layers:
                        sparse_feed = {}
                        for fc in fc_layers.values():
                            if fc.is_sparse():
                                sparse_feed.update(fc.feed_dict())
                        sess.run(update_representation, feed_dict=sparse_feed)

                    sess.run(train_loop, feed_dict={lr_placeholder: scheduler(e, opts)})
                    last = sess.run(dequeue_train_outfeed)

                    steps = 1 + i + e*opts.steps_per_epoch
                    batches_processed = batches_per_step*steps
                    for name, fc in fc_layers.items():
                        if fc.is_sparse():
                            logger.info(f"Average weights for layer {name}: {np.mean(last[name+'_non_zeros'][0])}")
                            for slot_name in fc.sparse_slots:
                                logger.info(f"Average {slot_name} for layer {name} : {np.mean(last[name+f'_{slot_name}'][0])}")
                            if not opts.disable_pruning:
                                logger.info(f"Starting prune and grow for layer {name}")
                                t0 = time.perf_counter()
                                prune_and_grow(name, fc, last, random_gen, steps, total_steps, opts)
                                t1 = time.perf_counter()
                                logger.info(f"Prune and grow for layer {name} complete in {t1-t0:0.3f} seconds")

                    if opts.log:
                        log_file.write(f"{batches_processed} {last['acc'][0]}\n")
                    progress.set_description(f"Loss {last['mean_loss'][0]:.5f} Accuracy {last['acc'][0]:.5f}")

            logger.info(f"Saving...")
            saver.save(sess, os.path.join(logpath, 'model.ckpt'))

        if opts.test_mode in ["all", "tests"]:
            test_feed = {}
            for fc in fc_layers.values():
                test_feed.update(fc.feed_dict())

            logger.info(f"Testing...")
            sess.run(metrics_initializer)
            sess.run(infeed_test_queue.initializer, feed_dict={
                     place_x: x_test_flat, place_y: y_test})
            sess.run(test_loop, feed_dict=test_feed)
            result = sess.run(dequeue_test_outfeed)

            test_loss = result['mean_loss'][-1]
            test_acc = result['acc'][-1]
            logger.info(f"Test loss: {test_loss:.8f} Test accuracy: {test_acc:.8f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and Test the simple Tensorflow model with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--test-mode",
        choices=['training', 'tests', 'all'],
        default="all",
        help="Use this flag to run the model in either 'training' or 'tests' mode or both ('all')")
    parser.add_argument("--densities", nargs=2, type=float, default=[0.0075, 1],
                        metavar=('density1', 'density2'),
                        help="Densities for the two FC layers. If density == 1 then "
                        "that layer will be a regular dense layer (not a popsparse layer).")
    parser.add_argument("--droprate", type=float, default=0.1,
                        help="Dropout rate (after first FC layer only).")
    parser.add_argument("--log", type=str, default=None,
                        help="File name for logging results.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set the random seed for numpy.")
    parser.add_argument("--checkpoint-path", type=str, default='checkpoints',
                        help="Path for saving the model.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Set the mini-batch size.")
    parser.add_argument("--steps-per-epoch", type=int, default=5,
                        help="How many times the IPU will return to host each epoch. If the "
                             "optimiser is rigl this will be the update rate for the "
                             "sparsity pattern.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train for.")
    parser.add_argument("--log-level", type=str, default='INFO',
                        choices=['NOTSET', 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    parser.add_argument("--disable-pruning", action='store_true',
                        help="By default training will update the sparse layers by pruning "
                        "--prune-ratio weights and then add new ones using the --regrow "
                        "strategy specified. Disabling pruning will train using a fixed "
                        "sparsity pattern instead.")
    parser.add_argument("--regrow", type=str, default='rigl',
                        choices=['random', 'rigl'],
                        help="Set the strategy used to re-grow pruned weights.")
    parser.add_argument("--prune-ratio", type=float, default=0.3,
                        help="Proportion of each layer's non-zero values to prune and replace at each Rig-L update.")
    parser.add_argument("--records-path", type=str, default=None,
                        help="If specified masks and weights will be saved to this path.")
    parser.add_argument("--optimizer", type=str, default="Momentum",
                        choices=["GradientDescent", "Momentum", "Adam"],
                        help="Which optimizer to use.")
    parser.add_argument("--data-type", type=str,
                        help="Choose the floating point type for the GRU cell's weights.",
                        choices=['fp32', 'fp16'], default='fp32')

    def parse_optimizer_arg(arg: str):
        name, value = arg.split('=')
        return (name, json.loads(value))

    parser.add_argument("--optimizer-arg", type=parse_optimizer_arg, action="append",
                        help="Extra argument for the chosen optimizer of the form argname=value. "
                        "Example: `use_nesterov=false`. "
                        "Can be input multiple times.")
    opts = parser.parse_args()

    logging.basicConfig(
        level=logging.getLevelName(opts.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    run_mnist(opts)
