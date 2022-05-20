# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from tqdm import tqdm
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import utils, ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu import rand_ops, pipelining_ops
from tensorflow.python.ipu.config import IPUConfig, StochasticRoundingBehaviour
import argparse
import json
import time
import os
from datetime import datetime
from ipu_sparse_ops import layers, optimizers, sparse_training
import logging
import wandb

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


def make_stages(
        fc_layers,
        droprate,
        opt_cls,
        opt_kws,
        iterations_per_dense_grad,
        training: bool,
        disable_dense_grad,
        png_queue):

    dense_grad_enabled = training and (not disable_dense_grad)

    def stage_1(lr, inputs, labels):
        # Gen counter to keep track of last-iteration for dense-gradient computation
        with tf.variable_scope("counter", reuse=tf.AUTO_REUSE, use_resource=True):
            itr_counter = tf.get_variable(
                "iterations", shape=[], dtype=tf.int32,
                trainable=False,
                initializer=tf.zeros_initializer())
            inc = tf.assign_add(itr_counter, 1)
            mod_itrs = tf.math.floormod(inc, iterations_per_dense_grad)
            last_itr = tf.equal(mod_itrs, 0)

        fc1 = fc_layers['fc1']
        relu1 = fc1(inputs, dense_grad_enabled and last_itr)

        # Use the IPU optimised version of dropout:
        if training:
            drop1 = rand_ops.dropout(relu1, rate=droprate)
        else:
            drop1 = relu1

        return lr, labels, drop1, last_itr

    def stage_2(lr, labels, drop1, last_itr):
        fc2 = fc_layers['fc2']
        relu2 = fc2(drop1, dense_grad_enabled and last_itr)

        with tf.variable_scope("metrics", reuse=tf.AUTO_REUSE, use_resource=True):
            acc, acc_op = tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(
                    relu2, axis=1, output_type=tf.dtypes.int32),
                name="accuracy")
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=tf.cast(relu2, dtype=tf.float32, name="logits_to_fp32"))
        with tf.control_dependencies([acc_op]):
            mean_loss = tf.reduce_mean(loss, name='mean_loss')

        return {'lr': lr, 'mean_loss': mean_loss, 'acc': acc_op, 'last_itr': last_itr}

    def optimizer_function(outputs):
        with tf.variable_scope("training", reuse=tf.AUTO_REUSE, use_resource=True):
            optimizer = optimizers.SparseOptimizer(opt_cls)(
                learning_rate=outputs['lr'], **opt_kws, name='optimise',
                sparse_layers=fc_layers.values(),
                dense_gradient_condition=outputs['last_itr'] if dense_grad_enabled else None,
                prune_and_grow_outfeed=png_queue)
        return pipelining_ops.OptimizerFunctionOutput(optimizer, outputs['mean_loss'])

    return [stage_1, stage_2], optimizer_function


def model(
        fc_layers,
        droprate,
        lr,
        opt_cls,
        opt_kws,
        iterations_per_dense_grad,
        training: bool,
        disable_dense_grad,
        last_outqueue,
        inputs,
        labels,
        png_queue):
    stages, optimizer_fn = make_stages(
        fc_layers=fc_layers,
        droprate=droprate,
        opt_cls=opt_cls,
        opt_kws=opt_kws,
        iterations_per_dense_grad=iterations_per_dense_grad,
        training=training,
        disable_dense_grad=disable_dense_grad,
        png_queue=png_queue)

    inputs = (lr, inputs, labels)
    for stage in stages:
        inputs = stage(*inputs)
    outputs = inputs
    loss = outputs['mean_loss']
    acc_op = outputs['acc']

    control_ops = []
    if training:
        output = optimizer_fn(outputs)
        train_op = output.opt.minimize(output.loss)
        control_ops.append(train_op)

        with tf.control_dependencies([acc_op]):
            mean_loss = tf.reduce_mean(loss, name='train_loss')
    else:
        with tf.control_dependencies([acc_op]):
            mean_loss = tf.reduce_mean(loss, name='test_loss')

    with tf.control_dependencies(control_ops):
        return last_outqueue.enqueue(outputs)


def make_bound_model(
        fc_layers,
        opts,
        lr_placeholder,
        opt_cls,
        opt_kws,
        train_batches_per_step,
        test_batches_per_step,
        train_queues,
        test_queues,
        png_queue,
        disable_dense_grad: bool = False):
    outfeed_train_queue, infeed_train_queue = train_queues
    outfeed_test_queue, infeed_test_queue = test_queues

    def bound_train_model(inputs, labels):
        return model(
            fc_layers=fc_layers,
            droprate=opts.droprate,
            lr=lr_placeholder,
            opt_cls=opt_cls,
            opt_kws=opt_kws,
            iterations_per_dense_grad=train_batches_per_step,
            training=True,
            disable_dense_grad=disable_dense_grad,
            last_outqueue=outfeed_train_queue,
            inputs=inputs,
            labels=labels,
            png_queue=png_queue)

    def bound_train_loop():
        return loop_builder(
            iterations=train_batches_per_step,
            builder_func=bound_train_model,
            infeed=infeed_train_queue)

    def bound_test_model(inputs, labels):
        return model(
            fc_layers=fc_layers,
            droprate=opts.droprate,
            lr=tf.convert_to_tensor(0.0, name="dummy_lr"),
            opt_cls=opt_cls,
            opt_kws=opt_kws,
            iterations_per_dense_grad=test_batches_per_step,
            training=False,
            disable_dense_grad=disable_dense_grad,
            last_outqueue=outfeed_test_queue,
            inputs=inputs,
            labels=labels,
            png_queue=png_queue)

    def bound_test_loop():
        return loop_builder(
            iterations=test_batches_per_step,
            builder_func=bound_test_model,
            infeed=infeed_test_queue)

    return [bound_train_loop, bound_test_loop], []


def make_bound_model_pipelining(
        fc_layers,
        opts,
        lr_placeholder,
        opt_cls,
        opt_kws,
        train_batches_per_step,
        test_batches_per_step,
        train_queues,
        test_queues,
        png_queue,
        disable_dense_grad: bool = False):
    outfeed_train_queue, infeed_train_queue = train_queues
    outfeed_test_queue, infeed_test_queue = test_queues

    def bound_train_loop(lr):
        stages, optimizer_fn = make_stages(
            fc_layers,
            opts.droprate,
            opt_cls,
            opt_kws,
            training=True,
            disable_dense_grad=disable_dense_grad,
            iterations_per_dense_grad=train_batches_per_step,
            png_queue=png_queue)

        return pipelining_ops.pipeline(
            computational_stages=stages,
            gradient_accumulation_count=opts.gradient_accumulation_count,
            repeat_count=train_batches_per_step,
            inputs=[lr],
            device_mapping=[0, 0],
            infeed_queue=infeed_train_queue,
            outfeed_queue=outfeed_train_queue,
            optimizer_function=optimizer_fn,
            offload_weight_update_variables=False,
            outfeed_loss=False,
            pipeline_schedule=next(p for p in pipelining_ops.PipelineSchedule
                                   if opts.pipeline_schedule == p.name),
            name="Pipeline_Train")

    def bound_test_loop():
        stages, _ = make_stages(
            fc_layers,
            opts.droprate,
            opt_cls,
            opt_kws,
            training=False,
            disable_dense_grad=disable_dense_grad,
            iterations_per_dense_grad=test_batches_per_step,
            png_queue=png_queue)

        return pipelining_ops.pipeline(
            computational_stages=stages,
            gradient_accumulation_count=opts.gradient_accumulation_count,
            repeat_count=test_batches_per_step,
            inputs=tf.Variable(initial_value=0.0, name="dummy_lr"),
            device_mapping=[0, 0],
            infeed_queue=infeed_test_queue,
            outfeed_queue=outfeed_test_queue,
            optimizer_function=None,
            outfeed_loss=False,
            pipeline_schedule=next(p for p in pipelining_ops.PipelineSchedule
                                   if opts.pipeline_schedule == p.name),
            name="Pipeline_Validation")

    return [bound_train_loop, bound_test_loop], [lr_placeholder]


def create_fc_layers(opts, batch_shape, random_gen):
    h1 = opts.hidden_size
    h2 = 10
    batch_size = batch_shape[0]
    density1, density2 = opts.densities
    make_sparse = layers.SparseFcLayer.from_random_generator
    dtype = tf.float16 if opts.data_type == 'fp16' else tf.float32
    partialsType = 'half' if opts.partials_type == 'fp16' else 'float'
    logger.info(f"Partials type: {partialsType}")

    fc_layers = {}
    if density1 >= 1:
        fc_layers['fc1'] = layers.DenseFcLayer(h1, name='dense_fc',
                                               dtype=dtype, use_bias=True, relu=True)
    else:
        limit = np.sqrt(6/((batch_shape[1] + h1)*density1))
        glorot_uniform_gen = partial(random_gen.uniform, -limit, limit)
        indices_random_gen = np.random.default_rng(seed=opts.seed)
        options = {"metaInfoBucketOversizeProportion": 0.5, "partialsType": partialsType,
                   "sharedBuckets": not opts.disable_shared_buckets}
        fc_layers['fc1'] = make_sparse(h1, batch_shape, density1,
                                       block_size=opts.block_size,
                                       values_initialiser_gen=glorot_uniform_gen,
                                       indices_initialiser_gen=indices_random_gen,
                                       matmul_options=options,
                                       name='sparse_fc',
                                       dtype=dtype,
                                       use_bias=True, relu=True,
                                       pooling_type=opts.pooling_type)
    if density2 >= 1:
        fc_layers['fc2'] = layers.DenseFcLayer(h2, name='dense_classifier',
                                               dtype=dtype, use_bias=True, relu=False)
    else:
        limit = np.sqrt(6/((h1 + h2)*density2))
        glorot_uniform_gen = partial(random_gen.uniform, -limit, limit)
        indices_random_gen = np.random.default_rng(seed=opts.seed)
        options = {"metaInfoBucketOversizeProportion": 0.3,
                   "sharedBuckets": not opts.disable_shared_buckets}
        fc_layers['fc2'] = make_sparse(h2, [batch_size, h1], density2,
                                       block_size=1,  # Layer is too small to use larger blocks
                                       values_initialiser_gen=glorot_uniform_gen,
                                       indices_initialiser_gen=indices_random_gen,
                                       matmul_options=options,
                                       name='sparse_classifier',
                                       dtype=dtype,
                                       use_bias=True, relu=False,
                                       pooling_type="NONE")
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


def prune_and_grow(name, fc, prune_and_grow_outputs, random_gen,
                   step, total_steps, opts, metainfo):

    def cosine_prune_schedule(t, T, max_pruned):
        s = sparse_training.cosine_prune_function(t, T, opts.cosine_prune_schedule)
        logger.info(f"t/T: {t}/{T} max:{max_pruned} sched: {s}")
        return int(np.ceil(max_pruned * s))


    # Sync the layer's internal host-side state with prune_and_grow_outputs results
    # (both weights and slots need to be kept in sync):
    fc.sync_internal_representation(
        {"nz": prune_and_grow_outputs[fc.get_values_var().name]},
        {
            slot_name: prune_and_grow_outputs[slot_name]
            for slot_name in fc.sparse_slots
        },
        {"metainfo": metainfo})

    grad_w_name = fc.get_values_var().name.replace('nz_values:0', 'grad_w')
    grow_results = sparse_training.prune_and_grow(
        name=fc.name,
        triplets=fc.get_triplets(),
        shape=fc.get_shape(),
        spec=fc.weights.spec,
        max_non_zeros=fc.get_max_non_zeros(),
        slot_triplets=fc.extract_slot_triplets(),
        prune_schedule=partial(cosine_prune_schedule, t=step, T=total_steps),
        prune_ratio=opts.prune_ratio,
        grad_w=np.array(prune_and_grow_outputs[grad_w_name]),
        grow_method=opts.regrow,
        random_gen=np.random.default_rng(seed=opts.seed),
        ipu_pooling_type=fc.pooling_type)

    if grow_results is not None:
        try:
            fc.update_triplets(grow_results['gt'])
            fc.update_slots_from_triplets(grow_results['gs'])
        except:
            logger.info(f"Failed to update representation with triplets:\n{grow_results['gt'][0]}\n{grow_results['gt'][1]}\n{grow_results['gt'][2]}")
            logger.info(f"Non-zeros: {len(grow_results['gt'][0])}")
            logger.info(f"Layer spec: {fc.weights.spec}")
            raise

    if opts.records_path and name == 'fc1':
        # Save the first hidden layer's weight mask for later analysis:
        save_weights(opts, name, fc, step)

    return opts.prune_ratio * sparse_training.cosine_prune_function(step, total_steps, opts.cosine_prune_schedule)


def save_weights(opts, name, fc, step):
    if fc.is_sparse():
        os.makedirs(opts.records_path, exist_ok=True)
        filename = os.path.join(opts.records_path, f"weights_{name}_{step:06}")
        dense_matrix = fc.extract_dense()
        np.save(filename, dense_matrix)


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


def make_pixel_permutation_matrix(opts, image_shape):
    """
    Return a fixed permutation matrix P to apply to the flattened image vectors.
    This is equivalent to choosing a different ordering of the pixels in the
    input vectors, instead of the arbitrary order imposed by flattening row-major.

    The matrix is returned in one-hot encoded format for efficient row manipulation.
    """
    num_pixels = image_shape[0] * image_shape[1]
    if opts.permute_input == 'block':
        # This mode breaks the image into sqrt(block_size) x sqrt(block_size)
        # tiles. Each tile is aligned with the sparsity block structure so that
        # sparse blocks operate on tiles.

        # First we decide on the block shape
        height = 1
        while height*height < opts.block_size:
            height <<= 1
        block_shape = (height, opts.block_size//height)

        assert np.prod(block_shape) == opts.block_size
        assert (image_shape[0] % block_shape[0]) == 0
        assert (image_shape[1] % block_shape[1]) == 0

        num_blocks = (
            image_shape[0]//block_shape[0],
            image_shape[1]//block_shape[1])

        # Then we define the permutation
        permutation = np.empty(shape=num_blocks + block_shape, dtype=int)

        # Indexing grid for entire image
        row_indices = np.arange(image_shape[0])
        col_indices = np.arange(image_shape[1])
        row_indices, col_indices = np.meshgrid(row_indices, col_indices)

        # Global (inter) block index, Local (intra) block index
        G_i, L_i = np.divmod(row_indices, block_shape[0])
        G_j, L_j = np.divmod(col_indices, block_shape[1])

        permutation[G_i, G_j, L_i, L_j] = (
            # Global offset of entire block
            (block_shape[0]*image_shape[1]*G_i + block_shape[1]*G_j) +
            # Local offset between elements in block
            (image_shape[1]*L_i + L_j))

        permutation = permutation.reshape(-1)
        assert len(permutation) == len(np.unique(permutation))

        return permutation
    elif opts.permute_input == 'random':
        return np.random.RandomState(seed=opts.seed).permutation(num_pixels)
    else:
        return np.arange(num_pixels)


def run_mnist(opts):
    if opts.pipelining and opts.gradient_accumulation_count < 4:
        raise ValueError("Pipelining requires at least 4 gradient accumulation steps.")
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
    batch_size = opts.batch_size // opts.gradient_accumulation_count
    batch_shape = [batch_size, num_pixels]
    num_train = y_train.shape[0]
    num_test = y_test.shape[0]
    dtype = tf.float16 if opts.data_type == 'fp16' else tf.float32

    # Flatten the images and cast the labels:
    permutation = make_pixel_permutation_matrix(opts, image_shape)

    x_train_flat = x_train.astype(dtype.as_numpy_dtype()).reshape(-1, num_pixels)
    x_test_flat = x_test.astype(dtype.as_numpy_dtype()).reshape(-1, num_pixels)

    x_train_flat[:, ...] = x_train_flat[:, permutation]
    x_test_flat[:, ...] = x_test_flat[:, permutation]

    if opts.records_path:
        os.makedirs(opts.records_path, exist_ok=True)
        filename = os.path.join(opts.records_path, "pixel_permutation")
        np.save(filename, permutation)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Decide how to split epochs into loops up front:
    if opts.pipelining:
        logger.info(f"Pipelined: micro-batch-size: {batch_size} accumulation-count: {opts.gradient_accumulation_count}")
    batches_per_epoch = num_train // (batch_size * opts.gradient_accumulation_count)
    test_batches = num_test // (batch_size * opts.gradient_accumulation_count)

    batches_per_step = opts.batches_per_step_override
    if batches_per_step is None:
        batches_per_step = batches_per_epoch // opts.steps_per_epoch

    if not (batches_per_epoch % opts.steps_per_epoch) == 0:
        raise ValueError(f"IPU steps per epoch {opts.steps_per_epoch} must divide batches per epoch {batches_per_epoch} exactly.")

    # Create FC layer descriptions:
    fc_layers = create_fc_layers(opts, batch_shape, random_gen)
    for name, fc in fc_layers.items():
        logger.info(f"Layer Config: {name}: {type(fc)}")

    # Put placeholders on the CPU host:
    with tf.device("cpu"):
        lr_placeholder = tf.placeholder(dtype, shape=[])

    # Create dataset and IPU feeds:
    def make_generator(features, labels):
        return lambda: zip(features, labels)

    # Input pipeline
    def make_dataset(features, labels, is_training: bool):
        dataset = tf.data.Dataset.from_generator(
            generator=make_generator(features, labels),
            output_types=(features.dtype, labels.dtype),
            output_shapes=(features.shape[1:], labels.shape[1:]))

        if is_training:
            dataset = dataset.shuffle(buffer_size=num_train, seed=opts.seed).cache()

        dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
        return dataset

    train_dataset = make_dataset(
        features=x_train_flat,
        labels=y_train,
        is_training=True)

    test_dataset = make_dataset(
        features=x_test_flat,
        labels=y_test,
        is_training=False)

    infeed_train_queue = ipu_infeed_queue.IPUInfeedQueue(train_dataset)
    outfeed_train_queue = ipu_outfeed_queue.IPUOutfeedQueue()
    outfeed_prune_and_grow_queue = ipu_outfeed_queue.IPUOutfeedQueue()
    infeed_test_queue = ipu_infeed_queue.IPUInfeedQueue(test_dataset)
    outfeed_test_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    # Get optimiser
    opt_cls, opt_kws = build_optimizer(opts.optimizer, opts.optimizer_arg)
    logger.info('Optimiser %s, optimiser keywords %s', opt_cls.__name__, opt_kws)

    # Get the bound model functions
    bound_model_fn = make_bound_model_pipelining if opts.pipelining else make_bound_model
    (bound_train_loop, bound_test_loop), train_inputs = bound_model_fn(
        fc_layers=fc_layers,
        opts=opts,
        lr_placeholder=lr_placeholder,
        opt_cls=opt_cls,
        opt_kws=opt_kws,
        train_batches_per_step=batches_per_step,
        test_batches_per_step=test_batches,
        train_queues=(outfeed_train_queue, infeed_train_queue),
        test_queues=(outfeed_test_queue, infeed_test_queue),
        png_queue=outfeed_prune_and_grow_queue,
        disable_dense_grad=opts.disable_dense_grad_override)

    # Use the bound builder functions to place the model on the IPU:
    with scopes.ipu_scope("/device:IPU:0"):
        train_loop = ipu_compiler.compile(bound_train_loop, inputs=train_inputs)
        test_loop = ipu_compiler.compile(bound_test_loop)

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
    utils.move_variable_initialization_to_cpu()
    config = IPUConfig()
    config.auto_select_ipus = 1

    if opts.on_demand:
        config.device_connection.enable_remote_buffers = True
        config.device_connection.type = utils.DeviceConnectionType.ON_DEMAND

    config.floating_point_behaviour.inv = False
    config.floating_point_behaviour.div0 = False
    config.floating_point_behaviour.oflo = False
    config.floating_point_behaviour.esr = StochasticRoundingBehaviour.ON
    config.floating_point_behaviour.nanoo = False
    config.configure_ipu_system()

    # These allow us to retrieve the results of IPU feeds:
    dequeue_test_outfeed = outfeed_test_queue.dequeue()
    dequeue_train_outfeed = outfeed_train_queue.dequeue()

    # Add dense gradient outfeed if we have sparse layers
    dequeue_prune_and_grow_outfeed = None
    if not opts.disable_dense_grad_override and any(fc.is_sparse() for fc in fc_layers.values()):
        dequeue_prune_and_grow_outfeed = outfeed_prune_and_grow_queue.dequeue()

    logger.info(f"Image shape: {image_shape} Training examples: {num_train} Test examples: {num_test}")
    logger.info(f"Epochs: {opts.epochs} Batch-size: {batch_size} Steps-per-epoch: {opts.steps_per_epoch} Batches-per-step: {batches_per_step}")
    total_steps = opts.steps_per_epoch * opts.epochs
    logger.info(f"Total steps: {total_steps}")

    if opts.log:
        # Open log and write header fields:
        log_file = open(opts.log, 'w')
        d1, d2 = opts.densities
        log_file.write(f"Iteration Density_{d1}_{d2}\n")

    if opts.restore:
        logpath = os.path.join(opts.checkpoint_path, opts.restore)
    else:
        logpath = os.path.join(opts.checkpoint_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.FileWriter(logpath)

    if opts.records_path:
        # Save the first hidden layer's weight mask for later analysis:
        save_weights(opts, 'fc1', fc_layers['fc1'], 0)

    # Run the model:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(infeed_train_queue.initializer)

        if opts.restore:
            saver.restore(sess, logpath + '/model.ckpt')

        if opts.test_mode in ["all", "training"]:
            logger.info(f"Training...")
            start = opts.start_epoch if opts.restore else 0
            progress = tqdm(
                range(start, opts.epochs),
                bar_format='{desc} Epoch: {n_fmt}/{total_fmt} {bar}')
            for e in progress:
                for i in range(opts.steps_per_epoch):
                    sess.run(metrics_initializer)

                    t1 = time.perf_counter()
                    sess.run(train_loop, feed_dict={lr_placeholder: scheduler(e, opts)})
                    t2 = time.perf_counter()
                    sess_time = t2 - t1
                    batch_time = sess_time / batches_per_step
                    throughput = batch_size / batch_time
                    logger.info(f"Time for sess.run: {sess_time:0.3f} "
                                f"Time per batch: {batch_time:0.6f} "
                                f"Throughput: {throughput}")

                    if opts.single_train_step_only:
                        return

                    train_outputs = sess.run(dequeue_train_outfeed)
                    if opts.pipelining:
                        train_outputs = train_outputs[-1]

                    # Get the last value for all items:
                    for k, v in train_outputs.items():
                        train_outputs[k] = v[-1]
                    logger.debug(f"Train outputs: {train_outputs.keys()}")

                    # Merge prune and grow fetches with last fetches:
                    if dequeue_prune_and_grow_outfeed is not None:
                        png_data = sess.run(dequeue_prune_and_grow_outfeed)
                        for k in png_data:
                            png_data[k] = png_data[k][-1]
                        logger.debug(f"Prune and grow outputs: {png_data.keys()}")

                    steps = 1 + i + e * opts.steps_per_epoch
                    batches_processed = batches_per_step*steps
                    for name, fc in fc_layers.items():
                        if fc.is_sparse():
                            var_name = fc.get_values_var().name
                            logger.info(f"Average weights for layer {name}: {np.mean(png_data[var_name])}")
                            for slot_name in fc.sparse_slots:
                                logger.info(f"Average {slot_name} for layer {name} : {np.mean(png_data[slot_name])}")
                            if i == 0 and e == opts.start_epoch:
                                metainfo = sess.run(fc.get_metainfo_var())
                            else:
                                metainfo = None
                            if not opts.disable_pruning:
                                logger.info(f"Starting prune and grow for layer {name}")
                                t0 = time.perf_counter()
                                prune_sched = prune_and_grow(name, fc, png_data, random_gen, steps, total_steps,
                                                             opts, metainfo=metainfo)
                                t1 = time.perf_counter()
                                logger.info(f"Prune and grow for layer {name} complete in {t1-t0:0.3f} seconds")
                                logger.info(f"Pruned proportion: {prune_sched}")
                                if opts.use_wandb:
                                    wandb.log({'Prune Schedule': prune_sched}, commit=False)

                    if opts.log:
                        log_file.write(f"{batches_processed} {train_outputs['acc']}\n")
                    if opts.use_wandb:
                        wandb.log({'Loss': train_outputs['mean_loss'], 'Accuracy': train_outputs['acc'],
                                   'Throughput': throughput}, commit=True)
                    progress.set_description(f"Loss {train_outputs['mean_loss']:.5f} Accuracy {train_outputs['acc']:.5f}")

                    # Only need to feed an updated sparsity representation if we are running rig-L:
                    if not opts.disable_pruning:
                        # Merge the feeds needed for all layers:
                        sparse_feed = {}
                        for fc in fc_layers.values():
                            if fc.is_sparse():
                                sparse_feed.update(fc.feed_dict())
                        sess.run(update_representation, feed_dict=sparse_feed)

                if e % opts.checkpoint_freq == 0:
                    logger.info(f"Saving...")
                    saver.save(sess, os.path.join(logpath, 'model.ckpt'))

        if opts.test_mode in ["all", "tests"]:
            logger.info(f"Testing...")
            sess.run(metrics_initializer)
            sess.run(infeed_test_queue.initializer)
            sess.run(test_loop)
            result = sess.run(dequeue_test_outfeed)

            test_loss = result['mean_loss'][-1]
            test_acc = result['acc'][-1]
            logger.info(f"Test loss: {test_loss:.8f} Test accuracy: {test_acc:.8f} Name: {opts.log}")
            if opts.use_wandb:
                wandb.run.summary["Test Loss"] = test_loss
                wandb.run.summary["Test Accuracy"] = test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and Test the simple TensorFlow model with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--test-mode",
        choices=['training', 'tests', 'all'],
        default="all",
        help="Use this flag to run the model in either 'training' or 'tests' mode or both ('all')")
    parser.add_argument("--densities", nargs=2, type=float, default=[0.01, 1],
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
    parser.add_argument("--checkpoint-freq", type=int, default=1, help="Frequency at which "
                        "to save checkpoints, in epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Set the global batch size.")
    parser.add_argument("--gradient-accumulation-count", type=int, default=1,
                        help="Number of steps to accumulate the gradient over in pipelining.")
    parser.add_argument("--pipelining", action="store_true",
                        help="Whether to enable pipelining, defaults to False.")
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
    parser.add_argument("--restore", default=None, type=str, help="If set, the checkpoint with the specified "
                        "name will be restored")
    parser.add_argument("--start-epoch", type=int, default=None, help="If restoring, specifies the starting "
                        "epoch to resume at, for the learning rate and pruning schedules.")
    parser.add_argument("--data-type", type=str,
                        help="Choose the floating point type for the weights.",
                        choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument("--block-size", type=int,
                        help="Set size of the square non-zero blocks in the sparse weights.",
                        choices=[1, 4, 8, 16], default=1)
    parser.add_argument("--partials-type", type=str,
                        help="Choose the floating point type for intermediate values.",
                        choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument("--pooling-type", default='NONE', choices=['NONE', 'AVG', 'MAX', 'SUM'],
                        help='Selects ipu-side dense gradient pooling method when block sparsity is enabled')
    parser.add_argument("--disable-shared-buckets", action="store_true",
                        help="Disable sparse matmul bucket sharing between the forward and backward passes")
    parser.add_argument("--permute-input", default='none', type=str,
                        help="Apply a non-trainable permutation to the input vectors that is fixed for all "
                             "train and test samples. This is used to avoid the arbitray block connection "
                             "structure induced by regular flattening of the input which otherwise hurts "
                             "performace for larger block sizes.",
                             choices=['none', 'block', 'random'])
    parser.add_argument("--hidden-size", type=int, default=320,
                        help="Set the hidden size of the FC network.")
    parser.add_argument("--batches-per-step-override", type=int, default=None,
                        help="Optional override for batches-per-step to assist in profiling.")
    parser.add_argument("--disable-dense-grad-override", action='store_true',
                        help="Optional override to turn off everything related to the dense gradient.")
    parser.add_argument("--single-train-step-only", action='store_true',
                        help="Only run a single step to assist in profiling.")
    pipeline_schedule_options = [p.name for p in pipelining_ops.PipelineSchedule]
    parser.add_argument('--pipeline-schedule', type=str, default="Grouped",
                        choices=pipeline_schedule_options,
                        help="Pipelining scheduler. In the 'Grouped' schedule the forward passes"
                        " are grouped together, and the backward passes are grouped together. "
                        "With 'Interleaved' the forward and backward passes are interleaved. "
                        "'Sequential' mimics a non-pipelined execution.")
    parser.add_argument("--cosine-prune-schedule", type=json.loads,
                        default={
                            'zero_steps': 0,
                            'phase_delay': 0,
                            'period': 0.5
                            },
                        help="Fine grained control of the pruning schedule.")
    parser.add_argument("--on-demand", action="store_true", help="Defer IPU attach until execution.")
    parser.add_argument("--use-wandb", action="store_true", help="Exports results to Weights and Biases for experiments tracking")
    parser.add_argument("--wandb-project-name", type=str, default=None, help="The name of the wandb project")
    parser.add_argument("--wandb-tags", type=str, nargs='+', default=None,
                        help="Tags to use for the current run in wandb. Can be used in the dashboard for sorting runs.")
    parser.add_argument("--wandb-name", type=str, default=None, help="A name for this run which will be used in wandb.")

    def parse_optimizer_arg(arg: str):
        name, value = arg.split('=')
        return (name, json.loads(value))

    parser.add_argument("--optimizer-arg", type=parse_optimizer_arg, action="append",
                        help="Extra argument for the chosen optimizer of the form argname=value. "
                        "Example: `use_nesterov=false`. "
                        "Can be input multiple times.")
    opts = parser.parse_args()

    if opts.restore and not opts.start_epoch:
        raise Exception("If restoring from a checkpoint, you must specify which epoch "
                        "to resume training from, using --start-epoch")

    logging.basicConfig(
        level=logging.getLevelName(opts.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    if opts.use_wandb:
        if opts.wandb_project_name is None:
            parser.error("If you set --use-wandb you must set --wandb-project-name")
        # Gather some important env variables to store in the info field:
        env_keys = ['POPLAR_SDK_ENABLED', 'POPLAR_ENGINE_OPTIONS', 'POPLAR_TARGET_OPTIONS', 'TF_POPLAR_FLAGS']
        env_keys += ['RDMAV_FORK_SAFE', 'POPLAR_LOG_LEVEL', 'POPLIBS_LOG_LEVEL']
        extra_info = ""
        for k in env_keys:
            extra_info += (f"{k}={os.getenv(k)}\n")
        wandb.init(name=opts.wandb_name, notes=extra_info,
                   project=opts.wandb_project_name, sync_tensorboard=True, tags=opts.wandb_tags)
        wandb.config.update(opts)

    run_mnist(opts)
