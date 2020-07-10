# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from tqdm import tqdm
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import utils, ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu import dataset_benchmark
from tensorflow.python.ipu import rand_ops
import argparse
import time
import os
from datetime import datetime
from ipu_sparse_ops import sparse, layers
import logging

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = logging.getLogger(os.path.basename(__file__))


def model(fc_layers, droprate, lr, iterations_per_step, training: bool,
          last_outqueue, inputs, labels):

    with tf.variable_scope("counter", reuse=tf.AUTO_REUSE, use_resource=True):
        itr_counter = tf.get_variable("iterations", shape=[], dtype=tf.int32,
                                      initializer=tf.zeros_initializer())
        mod_itrs = tf.math.floormod(itr_counter, iterations_per_step)
        last_itr = tf.equal(mod_itrs, 0)
        inc = tf.assign_add(itr_counter, 1)

    fc1 = fc_layers['fc1']
    with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE, use_resource=True):
        relu1 = fc1(inputs, last_itr)

    # Use the IPU optimised version of dropout:
    if training:
        drop1 = rand_ops.dropout(relu1, rate=droprate)
    else:
        drop1 = relu1

    fc2 = fc_layers['fc2']
    with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE, use_resource=True):
        relu2 = fc2(drop1, last_itr)

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
            momentum_slot_names = optimiser.get_slot_names()
            logger.debug(f"Optimiser slot names: {momentum_slot_names}")
            with tf.control_dependencies([train_op, acc_op]):
                mean_loss = tf.reduce_mean(loss, name='train_loss')
    else:
        with tf.control_dependencies([acc_op]):
            mean_loss = tf.reduce_mean(loss, name='test_loss')

    # Prepare results for feeds:
    last_results = {'mean_loss': mean_loss, 'acc': acc}
    for name, fc in fc_layers.items():
        if fc.is_sparse():
            with tf.variable_scope(name, reuse=True):
                weights_tensor = tf.convert_to_tensor(fc.get_values_var())
                last_results[name + '_non_zeros'] = weights_tensor
                if training:
                    dense_grad_w = fc.get_dense_grad_w(loss)
                    fc.record_momentum_var(optimiser, momentum_slot_names[0])
                    last_results[name + '_momentum'] = tf.convert_to_tensor(fc.momentum_var)
                    last_results[name + '_grad_w'] = tf.convert_to_tensor(dense_grad_w)

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

    fc_layers = {}
    if density1 >= 1:
        fc_layers['fc1'] = layers.DenseFcReluLayer(h1)
    else:
        limit = np.sqrt(6/(batch_shape[1] + h1))
        glorot_uniform_gen = partial(random_gen.uniform, -limit, limit)
        fc_layers['fc1'] = make_sparse(h1, batch_shape, density1, opts.prune_ratio,
                                       glorot_uniform_gen, opts.seed,
                                       bias=True, relu=True)
    if density2 >= 1:
        fc_layers['fc2'] = layers.DenseFcReluLayer(h2)
    else:
        limit = np.sqrt(6/(batch_shape[1] + h2))
        glorot_uniform_gen = partial(random_gen.uniform, -limit, limit)
        fc_layers['fc2'] = make_sparse(h2, [batch_size, h1], density2, opts.prune_ratio,
                                       glorot_uniform_gen, opts.seed,
                                       bias=True, relu=True)
    return fc_layers


def build_update_op(fc_layers):
    # Need to build update ops for each sparse layer so that
    # we can change sparsity pattern during training:
    fc_update_ops = {}
    for name, fc in fc_layers.items():
        if fc.is_sparse():
            with tf.variable_scope(name, reuse=True):
                fc_update_ops[name] = fc.update_sparsity_op()
            if not opts.disable_pruning:
                # If a layers sparsity pattern changed then its momentum
                # also need to be updated:
                fc_update_ops[name + '_momentum'] = fc.update_momentum_op()

    # Combine all layer updates into one update op:
    return tf.group(fc_update_ops.values())


def prune_and_grow(layer_name, layer, outputs_from_last_step, random_gen,
                   step, total_steps, opts):
    name = layer_name
    fc = layer
    last = outputs_from_last_step

    # Sync the layer's internal host-side state with last results
    # (both weights and momentum need to be kept in sync):
    fc.sync_internal_data(last[name+'_non_zeros'][0], last[name+'_momentum'][0])

    def bottom_k(a, k):
        return np.argpartition(a, k)[0:k]

    def cosine_prune_schedule(t, T, max_pruned):
        return int(np.ceil(.5 * max_pruned * (1 + np.cos(t * (np.pi/T)))))

    prune_count = cosine_prune_schedule(step, total_steps, fc.get_max_prune_count())
    logger.debug(f"Layer {name} pruning schedule: iteration {step} prune: {prune_count}")
    if prune_count <= 0:
        return

    # Convert to triplets to do the pruning:
    triplets = fc.extract_triplets()
    # We also need momentum triplets so that we can update the sparse momentum but we only
    # need to process the values because row and col indices must be the same:
    _, _, momentum_values = fc.extract_momentum_triplets()

    # Find the indices of the lowest magnitude weights:
    lowest_weight_idx = bottom_k(np.abs(triplets[2]), prune_count)
    logger.debug(f"Bottom {prune_count} value indices for layer {name}: {lowest_weight_idx}")
    logger.debug(f"Average trained triplet weights for layer {name}: {np.mean(triplets[2])}")
    prune_row_indices = triplets[0][lowest_weight_idx]
    prune_col_indices = triplets[1][lowest_weight_idx]
    logger.debug(f"Weight indices to prune: {name}: {list(zip(prune_row_indices, prune_col_indices))}")
    logger.debug(f"Weight values to prune: {name}: {triplets[2][lowest_weight_idx]}")
    if len(prune_row_indices) != prune_count:
        raise RuntimeError(f"Pruned {len(prune_row_indices)} indices but expected to prune {prune_count}")

    # Make new triplets with these indices removed:
    remaining_triplets = [np.delete(t, lowest_weight_idx) for t in triplets]
    # Prune the same indices from the momentum triplets:
    remaining_momentum_values = np.delete(momentum_values, lowest_weight_idx)

    if len(remaining_triplets[0]) + prune_count != len(triplets[0]):
        raise RuntimeError(f"Remaining index count {len(remaining_triplets[0])} is not the correct size: {fc.max_non_zeros}")

    # Get flat indices for the original index set:
    fc_shape = (fc.spec.input_size, fc.spec.output_size)
    original_flat_idx = np.ravel_multi_index((triplets[0], triplets[1]), fc_shape)

    # # Make the random initialiser for new values:
    # limit = np.sqrt(6/(fc.spec.input_size + fc.spec.output_size))
    # glorot_uniform_gen = partial(random_gen.uniform, -limit, limit)

    def zero_values(size=1):
        return [0]*size

    new_value_gen = zero_values

    if opts.regrow == 'rigl':
        # Grow back new indices using Rig-L: (https://arxiv.org/abs/1911.11134)
        grad_w = np.array(last[name+'_grad_w'][0][0])

        # We want to grow back weights at the positions with the highest gradient
        # magnitudes that are also not in the original set:
        abs_grad_flat = np.abs(grad_w.flatten())
        valid_new_idx = []
        argsorted = np.argsort(-abs_grad_flat)
        m = 1
        # Repeatedly take top-k in chunks until we have enough novel indices:
        while len(valid_new_idx) < prune_count:
            logger.debug(f"Next Top-k chunk {m}: {m*prune_count}")
            topk_flat_chunk_idx = argsorted[(m-1)*prune_count:m*prune_count]
            logger.debug(f"Next chunk indices: {topk_flat_chunk_idx}")
            mask = np.isin(topk_flat_chunk_idx, original_flat_idx, assume_unique=False)
            valid_new_idx = valid_new_idx + np.array(topk_flat_chunk_idx[~mask]).tolist()
            logger.debug(f"Valid so far: {valid_new_idx}")
            m += 1
            logger.debug(f"Valid new indices count: {len(valid_new_idx)}")

        topk_flat_idx = valid_new_idx[:prune_count]
        common = np.intersect1d(topk_flat_idx, original_flat_idx)
        if len(common):
            raise RuntimeError("Intersection of new and original indices must be empty.")
        logger.debug(f"Final non intersecting indices: {topk_flat_idx}")

        # Check the indices are unique before we use them:
        unique = np.unique(topk_flat_idx)
        duplicates = len(topk_flat_idx) - len(unique)
        if duplicates != 0:
            print(f"New indices contain {duplicates} duplicates:\n{topk_flat_idx}")
            raise RuntimeError("New indices are not unique")

        top_k_idx = np.unravel_index(topk_flat_idx, fc_shape)
        logger.debug(f"Layer {name} weight grad top-k indices: {top_k_idx}")
        new_triplets = (top_k_idx[0], top_k_idx[1], new_value_gen(size=prune_count))

    if opts.regrow == 'random':
        # Random replacement strategy: add back random indices
        # Gen some replacement random indices excluding all the existing
        # ones then we will swap for the pruned ones:
        new_triplets = sparse.random_triplets(fc.spec, seed=opts.seed, value_generator=new_value_gen,
                                              excluded_flat_indices=original_flat_idx, count=prune_count)

    # Join the triplets we kept with the newly generated ones:
    grown_rows = np.concatenate([remaining_triplets[0], new_triplets[0]]).astype(int)
    grown_cols = np.concatenate([remaining_triplets[1], new_triplets[1]]).astype(int)
    grown_values = np.concatenate([remaining_triplets[2], new_triplets[2]])
    # Momentum for new weights are set to zero:
    grown_momentum = np.concatenate([remaining_momentum_values, zero_values(size=prune_count)])

    if len(grown_rows) != fc.spec.max_non_zeros:
        raise ValueError(f"Grown row count {len(grown_rows)} does not match expected count {fc.spec.max_non_zeros}")
    if len(grown_cols) != fc.spec.max_non_zeros:
        raise ValueError(f"Grown col count {len(grown_cols)} does not match expected count {fc.spec.max_non_zeros}")
    if len(grown_values) != fc.spec.max_non_zeros:
        raise ValueError(f"Grown col count {len(grown_values)} does not match expected count {fc.spec.max_non_zeros}")
    if len(grown_momentum) != fc.spec.max_non_zeros:
        raise ValueError(f"Grown col count {len(grown_momentum)} does not match expected count {fc.spec.max_non_zeros}")

    # Now update the fc representations with the pruned and grown triplets:
    try:
        fc.update_triplets([grown_rows, grown_cols, grown_values])
        fc.update_momentum_from_triplets([grown_rows, grown_cols, grown_momentum])
    except:
        print(f"Old rows:\n{remaining_triplets[0]}")
        print(f"New rows:\n{new_triplets[0]}")
        print(f"Failed to update representation with triplets:\n{grown_rows}\n{grown_cols}\n{grown_values}")
        print(f"Non-zeros: {len(grown_rows)}")
        print(f"Layer spec: {fc.spec}")
        raise

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

    # Flatten the images and cast the labels:
    x_train_flat = x_train.astype(np.float32).reshape(-1, num_pixels)
    x_test_flat = x_test.astype(np.float32).reshape(-1, num_pixels)
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
        place_x = tf.placeholder(dtype=tf.float32, shape=data_shape, name="input")
        place_y = tf.placeholder(dtype=tf.int32, shape=[None], name="label")
        lr_placeholder = tf.placeholder(tf.float32, shape=[])
        for fc in fc_layers.values():
            fc.create_placeholders(tf.float32)

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

    # Use function binding to create all the builder functions that are neeeded:
    bound_train_model = partial(model, fc_layers, opts.droprate, lr_placeholder, batches_per_step,
                                True, outfeed_train_queue)
    bound_train_loop = partial(
        loop_builder, batches_per_step, bound_train_model, infeed_train_queue)
    bound_test_model = partial(model, fc_layers, opts.droprate, lr_placeholder, batches_per_step,
                               False, outfeed_test_queue)
    bound_test_loop = partial(loop_builder, test_batches,
                              bound_test_model, infeed_test_queue)

    # Use the bound builder functions to place the model on the IPU:
    with scopes.ipu_scope("/device:IPU:0"):
        train_loop = ipu_compiler.compile(bound_train_loop, inputs=[])
        test_loop = ipu_compiler.compile(bound_test_loop, inputs=[])
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

    # Merge the feeds needed for all layers:
    sparse_feed = {}
    for fc in fc_layers.values():
        sparse_feed.update(fc.feed_dict())

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
        # Must initialise the sparse layers separately:
        sess.run(update_representation, feed_dict=sparse_feed)

        if opts.test_mode in ["all", "training"]:
            logger.info(f"Training...")
            progress = tqdm(
                range(opts.epochs), bar_format='{desc} Epoch: {n_fmt}/{total_fmt} {bar}')
            for e in progress:
                for i in range(opts.steps_per_epoch):
                    sess.run(metrics_initializer)
                    # Only need to feed an updated sparsity representation if we are running rig-L:
                    if not opts.disable_pruning:
                        sess.run(update_representation, feed_dict=sparse_feed)
                    sess.run(train_loop, feed_dict={lr_placeholder: scheduler(e, opts)})
                    last = sess.run(dequeue_train_outfeed)

                    steps = 1 + i + e*opts.steps_per_epoch
                    batches_processed = batches_per_step*steps
                    for name, fc in fc_layers.items():
                        if fc.is_sparse():
                            logger.info(f"Average weights for layer {name}: {np.mean(last[name+'_non_zeros'][0])}")
                            logger.info(f"Average momentum for layer {name} : {np.mean(last[name+'_momentum'][0])}")
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
    opts = parser.parse_args()

    logging.basicConfig(
        level=logging.getLevelName(opts.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    run_mnist(opts)
