# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import json
import time
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from functools import partial

import wandb

import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import (
    utils,
    internal_ops,
    ipu_compiler,
    scopes,
    loops,
    pipelining_ops)
from tensorflow.python.ipu.ipu_infeed_queue import IPUInfeedQueue
from tensorflow.python.ipu.ipu_outfeed_queue import IPUOutfeedQueue
from tensorflow.python.ipu.config import IPUConfig

import tf_utils
from tf_utils import build_optimizer
import program_options
import grad_clip_opt
import scaling_opt
import global_step_update_opt
import data_utils
os.sys.path.append("../")  # dynamic_sparsity
from ipu_sparse_ops import optimizers, sparse, sparse_training, fp_slot_opt  # noqa: E402
from ipu_sparse_ops.transformer.transformer_dynsparse import DynsparseTransformer  # noqa: E402

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = logging.getLogger(os.path.basename(__file__))


def forward_pass(opts, transformer, iterations_per_step, is_training, outfeed, dense_queue, infeed):
    def make_counter():
        with tf.variable_scope("counter", reuse=tf.AUTO_REUSE, use_resource=True):
            itr_counter = tf.get_variable("iterations", [], tf.int32, trainable=False)
            increment_counter = tf.assign_add(itr_counter, 1)
            mod_itrs = tf.math.floormod(increment_counter, iterations_per_step)
            last_itr = tf.equal(mod_itrs, 0, name="last_update_itr")

            # Add accumulation counter if pipelined
            if opts.pipeline:
                grad_counter = internal_ops.get_current_iteration_counter()
                last_grad_itr = tf.equal(grad_counter, opts.gradient_accumulation_count-1, name="last_grad_itr")

                last_itr = tf.logical_and(last_itr, last_grad_itr, name="last_itr")

        return last_itr

    def make_src_mask(last_itr, source):
        with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE, use_resource=True):
            transformer.compute_dense_grad = last_itr
            autoregressive_mask = tf.constant(np.triu(np.ones([S, S], dtype=np.bool), k=1))
            source_mask = autoregressive_mask
            source_mask = tf.cast(source_mask, opts.dtype) * -10000
        return source_mask

    def loss_and_metrics(logits, source):
        with tf.variable_scope("metrics", reuse=tf.AUTO_REUSE, use_resource=True):
            # Implement autoregressice loss through teacher forcing
            # The first few tokens have no hope of being correct
            # so we exclude the first "offset" tokens from the loss
            offset = opts.autoregression_offset
            logits = tf.cast(logits[:, offset:-1], tf.float32)  # logits always full precision
            target = source[:, offset + 1:]
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

            # Accuracy
            acc, acc_op = tf.metrics.accuracy(target, predictions, name="token_accuracy")

            # Unweighted cross-entropy for tracking progress
            nll_loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=logits)
            nll_loss = tf.reduce_mean(nll_loss)
            perplexity = tf.exp(nll_loss)

            # Training loss (weighted cross-entropy)
            # the weight of the loss on each token is normalized by the number of
            # that token appears in the sequence
            # For instance if there are 10 padding tokens, the loss from each will have a weight of 1/10
            nll_weights = tf.expand_dims(target, -1)
            nll_weights = tf.equal(nll_weights, tf.transpose(nll_weights, perm=[0, 2, 1]))
            nll_weights = tf.cast(nll_weights, tf.float32)
            nll_weights = 1.0 / tf.reduce_sum(nll_weights, -1)
            training_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=target, logits=logits, weights=nll_weights)
            training_loss = tf.reduce_mean(training_loss)
        return {
            "training_loss": training_loss,
            "token_accuracy": acc,
            "acc_op": acc_op,
            "nll_loss": nll_loss,
            "perplexity": perplexity,
            "predictions": predictions,
            "target": target
        }

    def make_lr_schedule(global_step):
        with tf.variable_scope("training", reuse=tf.AUTO_REUSE, use_resource=True):
            # The learning rate schedule needs to be part of the graph so the lr can
            # change between different batchs within the same io step
            schedule = tf_utils.BertSchedule(opts, opts.dtype)
            lr = schedule(global_step)
        return lr

    def make_optimizer(lr, last_itr):
        with tf.variable_scope("training", reuse=tf.AUTO_REUSE, use_resource=True):
            optimizer_class, optimizer_kwargs = build_optimizer(opts.optimizer, opts.optimizer_arg)
            optimizer_class = optimizers.SparseOptimizer(optimizer_class)
            optimizer_class = global_step_update_opt.GlobalStepUpdateOptimizer(optimizer_class)
            if opts.loss_scale != 1:
                optimizer_class = scaling_opt.LossScalingOptimizer(optimizer_class)
                optimizer_kwargs['loss_scale'] = opts.loss_scale
                optimizer_kwargs['unscale_grad_pre_acc'] = opts.unscale_grad_pre_acc
            if opts.grad_acculation_mode == 'Avg':
                optimizer_class = scaling_opt.GradScalingOptimizer(optimizer_class)
                optimizer_kwargs['grad_scale'] = 1 / opts.gradient_accumulation_count
                optimizer_kwargs['scale_grad_pre_acc'] = opts.scale_grad_pre_acc
            if opts.grad_norm_clip:
                optimizer_class = grad_clip_opt.GradientClippingOptimizer(optimizer_class)
                optimizer_kwargs['norm_clip_threshold'] = opts.grad_norm_clip
            if opts.slots_fp_type is not None and tf.as_dtype(opts.slots_fp_type) != opts.dtype:
                optimizer_class = fp_slot_opt.SelectableSlotFPFormatOptimizer(optimizer_class)
                optimizer_kwargs['slots_dtype'] = opts.slots_fp_type
                optimizer_kwargs['force_fp32_weight_update'] = opts.force_fp32_weight_update
            optimizer = optimizer_class(learning_rate=lr, **optimizer_kwargs,
                                        sparse_layers=transformer.sparse_layers.values(),
                                        dense_gradient_condition=enable_dense_grad and last_itr,
                                        prune_and_grow_outfeed=dense_queue)
        return optimizer

    def make_pipeline_opt(outputs):
        optimizer = make_optimizer(outputs["learning_rate"], outputs["last_itr"])
        return pipelining_ops.OptimizerFunctionOutput(optimizer, outputs["training_loss"])

    def make_outfeed(lr, global_step, metrics, itr_counter):
        acc_op = metrics['acc_op']

        if is_training:
            with tf.control_dependencies([acc_op]):
                output_dict = {
                    **metrics,
                    "learning_rate": lr,
                    "global_step": tf.cast(global_step, tf.int32),
                    "iteration_counter": itr_counter}
                output = outfeed.enqueue(output_dict)
        else:
            # At inference time stream back the loss and accuracy
            with tf.control_dependencies([acc_op]):
                output = outfeed.enqueue(metrics)
        return output

    # Batch size and sequence length
    S = transformer.source_sequence_length
    enable_dense_grad = opts.prune_ratio is not None and opts.prune_ratio > 0

    if not opts.pipeline:
        # This autoregressive model is self-labeling needs only 1 input
        source = infeed
        last_itr = make_counter()
        source_mask = make_src_mask(last_itr, source)
        # Build the encoder
        logits = transformer.language_model(source=source, source_mask=source_mask,
                                            add_projection_layer=True, last_itr=last_itr,
                                            enable_dense_grad=enable_dense_grad,
                                            sparse_embeddings=opts.sparse_embeddings)
        metrics = loss_and_metrics(logits, source)
        if is_training:
            global_step = tf.cast(tf.train.get_or_create_global_step(), tf.int32)
            lr = make_lr_schedule(global_step)
            optimizer = make_optimizer(lr, last_itr)
            train_op = optimizer.minimize(metrics['training_loss'], global_step=global_step)
        else:
            lr, global_step = None, None
            train_op = tf.no_op()

        with tf.control_dependencies([train_op]):
            with tf.variable_scope("counter", reuse=tf.AUTO_REUSE, use_resource=True):
                itr_counter = tf.get_variable("iterations", [], tf.int32, trainable=False)
            output = make_outfeed(lr, global_step, metrics, itr_counter)
        return output
    else:
        def first_stage(global_step, source, input_stage_func):
            last_itr = make_counter()
            source_mask = make_src_mask(last_itr, source)
            return input_stage_func(source, source_mask, last_itr, global_step)

        def last_stage(encoder_out, source_mask, *args, **kwargs):
            last_itr = args[0]
            global_step = args[1]
            source = args[2]
            output_stage_func = kwargs['output_stage_func']
            logits, *_ = output_stage_func(encoder_out, source_mask, *args)
            metrics = loss_and_metrics(logits, source)
            if is_training:
                metrics.update({
                        "learning_rate": make_lr_schedule(global_step),
                        "last_itr": last_itr,
                        "global_step": tf.convert_to_tensor(global_step)
                })
                return metrics
            else:
                metrics['last_itr'] = last_itr
                return metrics

        stages, device_mapping, stage_options = transformer.language_model_stages(enable_dense_grad=enable_dense_grad,
                                                                                  sparse_embeddings=opts.sparse_embeddings)
        stages[0] = partial(first_stage, input_stage_func=stages[0])
        stages[-1] = partial(last_stage, output_stage_func=stages[-1])

        pipeline_op = pipelining_ops.pipeline(
            computational_stages=stages,
            gradient_accumulation_count=opts.gradient_accumulation_count,
            gradient_accumulation_dtype=opts.gradient_accumulation_dtype,
            repeat_count=iterations_per_step,
            inputs=[tf.cast(tf.train.get_or_create_global_step(), tf.int32)],
            infeed_queue=infeed,
            outfeed_queue=outfeed,
            optimizer_function=make_pipeline_opt if is_training else None,
            device_mapping=device_mapping,
            offload_activations=opts.offload_activations,
            offload_gradient_accumulation_buffers=opts.offload_gradient_accumulation_buffers,
            offload_weight_update_variables=opts.offload_weight_update_variables,
            forward_propagation_stages_poplar_options=stage_options,
            backward_propagation_stages_poplar_options=stage_options,
            name="Pipeline")

        return pipeline_op


def run_training(opts, transformer):
    # Construct the training graph
    training_graph = tf.Graph()
    with training_graph.as_default():
        with tf.device("cpu"):
            dataset, num_train, vocab = data_utils.make_dataset(opts, use_synthetic_data=opts.use_synthetic_data, training=True)

        # Calculate dataset length
        batch_size = opts.batch_size
        if opts.pipeline:
            batch_size *= opts.gradient_accumulation_count
        batches_per_epoch = num_train // batch_size
        io_steps_per_epoch = batches_per_epoch // opts.repeat_count
        total_io_steps = opts.nepochs * io_steps_per_epoch
        total_global_steps = opts.nepochs * io_steps_per_epoch * opts.repeat_count
        logger.info(f"Effective batch-size (global batch): {batch_size}, "
                    f"IO steps per epoch: {io_steps_per_epoch}, "
                    f"Total IO steps: {total_io_steps} "
                    f"Total global steps: {total_global_steps}")

        if opts.prune_ratio is not None and opts.prune_ratio > 0:
            # Compute the pruning ratio when the learning rate will reach a minimum
            lr_decay_steps = opts.cooldown_steps + opts.warmup_steps
            lr_min_epochs = lr_decay_steps / (io_steps_per_epoch * opts.repeat_count)
            remainining_prune_ratio = opts.prune_ratio * sparse_training.cosine_prune_function(lr_decay_steps, total_global_steps, opts.cosine_prune_schedule)
            logger.warn(f"\n\nThe learning rate schedule will reach a minimum after {lr_min_epochs:0.2f} epochs, "
                        f"at which point the pruning ratio will be {remainining_prune_ratio:0.3f}\n\n")
            logger.info(f"Cosine prune schedule options: {opts.cosine_prune_schedule}")

        logger.info("Creating infeed and outfeed queues")
        # Queues for streaming from host to device and back
        train_infeed = IPUInfeedQueue(dataset)
        train_outfeed = IPUOutfeedQueue()
        prune_and_grow_outfeed = IPUOutfeedQueue()

        # Helper function
        def loop_builder(iterations, builder_func, infeed):
            return loops.repeat(iterations, builder_func, [], infeed)

        # Compile the forward and backward pass for training
        with scopes.ipu_scope("/device:IPU:0"):
            if opts.pipeline:
                logger.info("Creating pipelined training graph")
                train_loop = partial(forward_pass, opts, transformer,
                                     opts.repeat_count, True,
                                     train_outfeed, prune_and_grow_outfeed, train_infeed)
            else:
                logger.info("Creating training graph")
                train_body = partial(forward_pass, opts, transformer,
                                     opts.repeat_count, True,
                                     train_outfeed, prune_and_grow_outfeed)
                train_loop = partial(loop_builder, opts.repeat_count, train_body, train_infeed)
            train_loop = ipu_compiler.compile(train_loop, inputs=[])
            transformer.buildSparsityUpdateOps()

        # Metrics
        with tf.device("cpu"):
            metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
            metrics_initializer = tf.variables_initializer(var_list=metrics_vars)
            saver = tf.train.Saver()

            # These ops are declared here so that the graph can be frozen afterwards
            global_initializer = tf.global_variables_initializer()
            train_outfeed_dequeue = train_outfeed.dequeue()
            if opts.prune_ratio is not None and opts.prune_ratio > 0:
                prune_and_grow_dequeue = prune_and_grow_outfeed.dequeue()
            utils.move_variable_initialization_to_cpu()

            # Tensorboard
            log_name = "logs/" + datetime.now().isoformat()
            summary_writer = tf.summary.FileWriter(logdir=os.path.join(opts.train_checkpoint_path, log_name),
                                                   flush_secs=5)

    # Run the model:
    training_graph.finalize()  # no more new ops added from here on out
    with tf.Session(graph=training_graph) as sess:
        logger.info(f"Initializing training session")
        sess.run(global_initializer)
        sess.run(train_infeed.initializer)
        logger.info(f"Training...")
        progress = tqdm(range(opts.nepochs))
        for e in progress:
            sess.run(metrics_initializer)
            for io_step in range(io_steps_per_epoch):
                # Train the model
                step_start_time = time.perf_counter()
                sess.run(train_loop)
                ipu_train_time = time.perf_counter() - step_start_time

                session_outputs = sess.run(train_outfeed_dequeue)[-1]
                logger.debug(f"Train outputs: {session_outputs.keys()}")

                # Calculate avg throughput
                num_tokens = transformer.source_sequence_length * opts.repeat_count * batch_size
                throughput = num_tokens / ipu_train_time

                # Log progress - average stats over the last accumulation step only:
                start_point = -1 if not opts.pipeline else -opts.gradient_accumulation_count
                lr = np.mean(session_outputs["learning_rate"][start_point:])
                training_loss = np.mean(session_outputs['training_loss'][start_point:])
                std_training_loss = np.std(session_outputs['training_loss'][start_point:])
                nll_loss = np.mean(session_outputs['nll_loss'][start_point:])
                perplexity = np.mean(session_outputs["perplexity"][start_point:])
                token_accuracy = np.mean(session_outputs['token_accuracy'][start_point:])
                global_step = session_outputs['global_step'][start_point:][-1]
                logger.info(
                    f"\nEpoch {e}: io_step {io_step+1}/{io_steps_per_epoch}"
                    f"\nGlobal step: {global_step}/{total_global_steps}"
                    f"\nTraining loss : {training_loss:.4f}"
                    f"\nTraining loss standard deviation: {std_training_loss:.4f}"
                    f"\nXentropy loss : {nll_loss:.4f}"
                    f"\nPerplexity : {perplexity:.3f}"
                    f"\nToken accuracy: {token_accuracy:.2f}"
                    f"\nLearning rate: {lr:3.4e}"
                    f"\nThroughput {throughput:.1f} token/s")

                if opts.decode and logger.level <= logging.INFO:
                    try:
                        text_pred, text_target = data_utils.decode_prediction(prediction=session_outputs['predictions'][-1],
                                                                              target=session_outputs['target'][-1],
                                                                              vocab=vocab)
                        logger.info(f"\nTarget: {text_target}\n\nPrediction: {text_pred}\n")
                    except Exception as ex:
                        logger.warn(f"Decoding failed: {ex}")

                summary_value = [tf.Summary.Value(tag="perplexity", simple_value=perplexity),
                                 tf.Summary.Value(tag="training_loss", simple_value=training_loss),
                                 tf.Summary.Value(tag="stddev_training_loss", simple_value=std_training_loss),
                                 tf.Summary.Value(tag="xentropy_loss", simple_value=nll_loss),
                                 tf.Summary.Value(tag="token_accuracy", simple_value=token_accuracy),
                                 tf.Summary.Value(tag="learning_rate", simple_value=lr),
                                 tf.Summary.Value(tag="throughput", simple_value=throughput),
                                 tf.Summary.Value(tag="epoch", simple_value=e)]

                # If we just completed the last io step we do not
                # prune and grow regardless, otherwise check the prune ratio:
                if io_step + 1 < io_steps_per_epoch and transformer.prune_ratio is not None and transformer.prune_ratio > 0:
                    # Retrieve p and g results from the conditional queue:
                    prune_and_grow_data = sess.run(prune_and_grow_dequeue)
                    for k in prune_and_grow_data:
                        prune_and_grow_data[k] = prune_and_grow_data[k][-1]
                    logger.debug(f"Prune and grow outputs: {prune_and_grow_data.keys()}")

                    prune_and_grow_time, cosine_schedule_factor = transformer.syncPruneAndRegrowOnHost(
                        opts.cosine_prune_schedule, global_step, total_global_steps, prune_and_grow_data)
                    transformer.streamSparsityFromHostToDevice()
                    summary_value.extend([tf.Summary.Value(tag="prune+grow_time", simple_value=prune_and_grow_time),
                                          tf.Summary.Value(tag="cosine_schedule_factor", simple_value=cosine_schedule_factor)])

                    for layer_name, sparse_layer in transformer.sparse_layers.items():
                        values_var = sparse_layer.get_values_var()
                        grad_w_name = values_var.name.replace('nz_values:0', 'grad_w')
                        grad_w = np.array(prune_and_grow_data[grad_w_name])
                        if (opts.log_histograms):
                            histogram = tf_utils.make_histogram_proto(grad_w, bins_count=opts.bins_count)
                            summary_value.extend([tf.Summary.Value(tag=layer_name + "/dense_grad_w", histo=histogram)])

                        summary_value.extend([tf.Summary.Value(tag=layer_name + "/dense_grad_w_stddev", simple_value=np.std(grad_w)),
                                              tf.Summary.Value(tag=layer_name + "/dense_grad_w_mean", simple_value=np.mean(grad_w)),
                                              tf.Summary.Value(tag=layer_name + "/dense_grad_w_min", simple_value=np.min(grad_w)),
                                              tf.Summary.Value(tag=layer_name + "/dense_grad_w_max", simple_value=np.max(grad_w))])

                        for slot_name, slot in sparse_layer.get_slot_var_dict().items():
                            slot_val = prune_and_grow_data[slot.tf_variable.name]
                            if opts.log_histograms:
                                histogram = tf_utils.make_histogram_proto(slot_val, bins_count=opts.bins_count)
                                summary_value.extend([tf.Summary.Value(tag=slot_name, histo=histogram)])
                            summary_value.extend([tf.Summary.Value(tag=slot_name + "/stddev", simple_value=np.std(slot_val)),
                                                  tf.Summary.Value(tag=slot_name + "/mean", simple_value=np.mean(slot_val)),
                                                  tf.Summary.Value(tag=slot_name + "/min", simple_value=np.min(slot_val)),
                                                  tf.Summary.Value(tag=slot_name + "/max", simple_value=np.max(slot_val))])

                # Log to tensorboard (outside any graph)
                summary = tf.Summary(value=summary_value)
                summary_writer.add_summary(summary, np.mean(global_step))
                if opts.use_wandb:
                    wandb.tensorflow.log(summary.SerializeToString())
                logger.info(f"Total time for step {time.perf_counter() - step_start_time}")
                logger.info(f"IPU train time for step {ipu_train_time}")

            logger.info(f"Saving model after epoch {e}")
            saver.save(sess, os.path.join(opts.train_checkpoint_path, 'model_' + str(e) + '.ckpt'))
            os.sys.stdout.flush()
        logger.info(f"Training complete.")


def run_testing(opts, transformer):
    testing_graph = tf.Graph()
    with testing_graph.as_default():
        with tf.device("cpu"):
            logger.info("Creating test dataset")
            dataset, num_test, vocab = data_utils.make_dataset(opts, use_synthetic_data=opts.use_synthetic_data, training=False)

            batch_size = opts.batch_size
            if opts.pipeline:
                batch_size *= opts.gradient_accumulation_count
            batches_per_epoch = num_test // batch_size
            logger.info(f"Effective batch-size (global batch): {batch_size}")

            logger.info("Creating infeed and outfeed queues")
            test_infeed = IPUInfeedQueue(dataset)
            test_outfeed = IPUOutfeedQueue()

        # Compile the forward pass for testing
        with scopes.ipu_scope("/device:IPU:0"):
            # Helper function
            def loop_builder(iterations, builder_func, infeed):
                return loops.repeat(iterations, builder_func, [], infeed)

            if opts.pipeline:
                logger.info("Creating pipelined test graph")
                test_loop = partial(forward_pass, opts, transformer,
                                    batches_per_epoch, False, test_outfeed,
                                    dense_queue=None, infeed=test_infeed)
            else:
                logger.info("Creating test graph")
                test_loop = partial(forward_pass, opts, transformer,
                                    batches_per_epoch, False, test_outfeed,
                                    None)
                test_loop = partial(loop_builder, batches_per_epoch, test_loop, test_infeed)
            test_loop = ipu_compiler.compile(test_loop, inputs=[])

        # Metrics
        with tf.device("cpu"):
            metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
            metrics_initializer = tf.variables_initializer(var_list=metrics_vars)
            saver = tf.train.Saver()

    if opts.restore_epoch is None:
        checkpoint = tf.train.latest_checkpoint(opts.train_checkpoint_path)
    else:
        checkpoint = opts.train_checkpoint_path + "/model_" + str(opts.restore_epoch) + ".ckpt"

    with tf.Session(graph=testing_graph) as sess:
        # The sparsity will also  be streamed from the checkpoint
        logger.info("Restoring weights")
        saver.restore(sess, checkpoint)
        sess.run(test_infeed.initializer)
        sess.run(metrics_initializer)

        # Run inference (whole dataset in one session call)
        logger.info("Testing...")
        dt = time.perf_counter()
        sess.run(test_loop)
        dt = time.perf_counter() - dt
        session_outputs = sess.run(test_outfeed.dequeue())[-1]

        # Test set performance
        # Log progress
        nll_loss = session_outputs['nll_loss'][-1]
        training_loss = session_outputs['training_loss'][-1]
        perplexity = session_outputs["perplexity"][-1]
        token_accuracy = session_outputs['token_accuracy'][-1]
        desc = (
            f"\nTraining loss : {training_loss:.4f}"
            f"\nXentropy loss : {nll_loss:.4f}"
            f"\nPerplexity : {perplexity:.3f}"
            f"\nToken accuracy: {token_accuracy:.2f}")
        logger.info(desc)

        if(opts.decode and opts.log_level == 'INFO'):
            text_pred, text_target = data_utils.decode_prediction(prediction=session_outputs['predictions'][-1],
                                                                  target=session_outputs['target'][-1],
                                                                  vocab=vocab)
            logger.info(f"Target: {text_target}\n"
                        f"Prediction: {text_pred}\n")
        os.sys.stdout.flush()

        logger.info(f"Test complete.")

    return desc


def run_language_model(opts):
    if opts.random_seed is not None:
        utils.reset_ipu_seed(opts.random_seed)

    # Setup and acquire an IPU device:
    logging.info("Acquiring devices")
    if not opts.pipeline:
        opts.num_shards = 1  # FIX-ME enable sparse models using multiple shards

    # Make sure that no matter the number of shards/stages required, we always
    # acquire a power of 2 ipus (else attachment will fail)
    k = 0
    while 2**k < opts.num_shards:
        k += 1
    num_ipus = 2**k
    logger.info(f"Need {opts.num_shards} IPUs, requesting {num_ipus}")
    config = IPUConfig()
    config.device_connection.enable_remote_buffers = True

    if opts.compile_only and opts.on_demand:
        raise ValueError("Can only provide one of --on-demand, --compile-only.")

    if opts.compile_only:
        if opts.compile_only_ipu_version is None:
            raise AttributeError(
                "Must provide --compile-only-ipu-version if --compile-only is set.")

        config.device_connection.version = opts.compile_only_ipu_version
        config.device_connection.type = utils.DeviceConnectionType.NEVER

    if opts.on_demand:
        config.device_connection.type = utils.DeviceConnectionType.ON_DEMAND

    config.auto_select_ipus = num_ipus
    config.allow_recompute = opts.recompute
    # Enable stochastic rounding
    config.floating_point_behaviour.inv = False
    config.floating_point_behaviour.div0 = False
    config.floating_point_behaviour.oflo = False
    config.floating_point_behaviour.esr = True
    config.floating_point_behaviour.nanoo = False
    config = sparse.set_system_config(config, custom_op_debug_printing=opts.debug_dense_grad)
    config.configure_ipu_system()

    transformer = DynsparseTransformer(opts)
    if opts.mode in ["all", "train"]:
        run_training(opts, transformer)

    if opts.mode in ["all", "test"]:
        run_testing(opts, transformer)


if __name__ == '__main__':
    # Parse arguments
    opts = program_options.get_program_options()

    # Set sync options
    if not opts.extra_poplar_options_disable:
        # Fetch the existing flags as json so we don't overwrite options set by the user
        engine_options = json.loads(os.getenv('POPLAR_ENGINE_OPTIONS', '{}'))
        target_options = json.loads(os.getenv('POPLAR_TARGET_OPTIONS', '{}'))
        logger.info('Received engine options %s and target options %s', engine_options, target_options)

        # Set options so that each IPU can sync independently with host if user hasn't already
        if opts.extra_poplar_options_sync_enable:
            if 'target.syncReplicasIndependently' not in engine_options:
                engine_options['target.syncReplicasIndependently'] = 'true'
            if 'syncConfiguration' not in target_options:
                target_options['syncConfiguration'] = 'ipuAndAll'

        # Set option to fix number of threads used for stream callbacks if user hasn't already
        if 'streamCallbacks.numWorkerThreads' not in engine_options:
            engine_options['streamCallbacks.numWorkerThreads'] = \
                opts.extra_poplar_options_num_callback_threads

        # Write flags back as json after editing them
        os.environ['POPLAR_ENGINE_OPTIONS'] = json.dumps(engine_options)
        os.environ['POPLAR_TARGET_OPTIONS'] = json.dumps(target_options)
        logger.info('Wrote engine options %s and target options %s', engine_options, target_options)

    if opts.use_wandb:
        # Gather some important env variables to store in the info field:
        env_keys = ['POPLAR_SDK_ENABLED', 'POPLAR_ENGINE_OPTIONS', 'POPLAR_TARGET_OPTIONS', 'TF_POPLAR_FLAGS']
        env_keys += ['RDMAV_FORK_SAFE', 'POPLAR_LOG_LEVEL', 'POPLIBS_LOG_LEVEL']
        extra_info = ""
        for k in env_keys:
            extra_info += (f"{k}={os.getenv(k)}\n")
        wandb.init(name=opts.wandb_name, notes=extra_info, project=opts.wandb_project_name, sync_tensorboard=True, tags=opts.wandb_tags)
        wandb.config.update(opts)

    logging.basicConfig(
        level=logging.getLevelName(opts.log_level),
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    logger.info(f"Model configuration: \n{opts}")
    run_language_model(opts)
