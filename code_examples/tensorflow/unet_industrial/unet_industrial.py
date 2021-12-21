# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
from tensorflow.python.ipu.config import IPUConfig
import numpy as np
import tensorflow.compat.v1 as tf
from functools import partial
from packaging import version
import time
import os
import random
from threading import Thread
from tensorflow.keras.layers import Activation, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.python import ipu


# Dice coefficient for loss function
def dice_coef_fn(y_true, y_pred):
    smooth = 1.
    y_true_flat = tf.layers.flatten(y_true)
    y_pred_flat = tf.layers.flatten(y_pred)
    intersection = tf.math.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.math.reduce_sum(y_true_flat) + tf.math.reduce_sum(y_pred_flat)
    return (2. * intersection + smooth) / (union + smooth)


# Dice Coef loss for 2-class segemntation
def dice_coef_loss_binary_fn(y_true, y_pred):
    dice_0 = dice_coef_fn(y_true, y_pred)
    dice_1 = dice_coef_fn(1. - y_true, 1. - y_pred)

    # with tf.control_dependencies([ipu.ops.internal_ops.print_tensor(dice_0, "print_dice_0"), ipu.ops.internal_ops.print_tensor(dice_1, "print_dice_1")]):
    mean_dice = (dice_0 + dice_1) / 2
    return 1. - mean_dice


# Build the model graph
def model_fn(mode=None, params=None, args=None, inference_features=None):
    tf_dtype = tf.float16 if args.dtype == 'float16' else tf.float32
    partials_type = np.float16 if args.partials_type == "float16" else np.float32


    # Chained convolutional layers with master weights
    def conv2d_chain(x, n_filters, name=''):
        for i, f in enumerate(n_filters):
            # Create variables for master weights (potentially in a higher precision)
            w = tf.get_variable("conv2d/kernel" + str(i) + name,
                                shape=[3, 3, x.shape[3], f],
                                dtype=args.master_weight_type,
                                trainable=True,
                                initializer=tf.initializers.variance_scaling(scale=2.0, distribution='uniform', mode='fan_in'))
            b = tf.get_variable("conv2d/bias" + str(i) + name,
                                shape=[f],
                                dtype=args.master_weight_type,
                                trainable=True,
                                initializer=tf.initializers.constant(0.0))
            # Add ops to the graph
            x = tf.nn.conv2d(x,
                             filters=tf.cast(w, tf_dtype),
                             padding='SAME')
            x = tf.nn.bias_add(x, tf.cast(b, tf_dtype), data_format="NHWC")
            x = tf.nn.relu(x)
        return x


    # Deconvolutional layer with master weights
    def conv2d_transpose(x, n_filters, name=''):
        # Create variables for master weights (potentially in a higher precision)
        w = tf.get_variable("conv2d_transpose/kernel" + name,
                            shape=[3, 3, n_filters, x.shape[3]],
                            dtype=args.master_weight_type,
                            trainable=True,
                            initializer=tf.initializers.variance_scaling(scale=2.0, distribution='uniform', mode='fan_in'))
        b = tf.get_variable("conv2d_transpose/bias" + name,
                            shape=[n_filters],
                            dtype=args.master_weight_type,
                            trainable=True,
                            initializer=tf.initializers.constant(0.0))
        # Add ops to the graph
        x = tf.nn.conv2d_transpose(x,
                                   filters=tf.cast(w, tf_dtype),
                                   strides=(2, 2),
                                   output_shape=[
                                       x.shape[0], x.shape[1]*2, x.shape[2]*2, n_filters],
                                   padding='SAME')
        x = tf.nn.bias_add(x, tf.cast(b, tf_dtype), data_format="NHWC")
        x = tf.nn.relu(x)
        return x


    # Pipeline stage: encoder
    def pipeline_stage1(global_step, features, labels):
        x = features
        skip_connections = []
        x = conv2d_chain(x, [32, 32], 'encoder0')
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)

        x = conv2d_chain(x, [32, 32], 'encoder1')
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)

        x = conv2d_chain(x, [64, 64], 'encoder2')
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)

        x = conv2d_chain(x, [128, 128], 'encoder3')
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)

        return tuple([global_step, x, labels] + skip_connections)


    # Pipeline stage: decoder
    def pipeline_stage2(global_step, x, labels, *skip_connections):
        skip_connections = [s for s in skip_connections]
        x = conv2d_chain(x, [256, 256], 'bottleneck')

        x = conv2d_transpose(x, 128, 'decoder1')
        x = tf.concat([x, skip_connections.pop()], axis=-1)
        x = conv2d_chain(x, [128, 64], 'decoder1')

        x = conv2d_transpose(x, 64, 'decoder2')
        x = tf.concat([x, skip_connections.pop()], axis=-1)
        x = conv2d_chain(x, [64, 32], 'decoder2')

        x = conv2d_transpose(x, 32, 'decoder3')
        x = tf.concat([x, skip_connections.pop()], axis=-1)
        x = conv2d_chain(x, [32, 16], 'decoder3')

        x = conv2d_transpose(x, 16, 'decoder4')
        x = tf.concat([x, skip_connections.pop()], axis=-1)
        x = conv2d_chain(x, [32, 32], 'decoder4')

        x = Conv2D(args.output_classes-1, (1, 1))(x)

        y_pred_logits = x
        y_pred = tf.math.sigmoid(y_pred_logits)
        predictions = tf.math.round(y_pred)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return predictions

        labels_fp32 = tf.cast(labels, tf.float32)
        y_pred_fp32 = tf.cast(y_pred, tf.float32)
        dice_coef = dice_coef_fn(labels_fp32, y_pred_fp32)
        dice_loss = 1 - dice_coef
        dice_coef_loss_binary = dice_coef_loss_binary_fn(
            labels_fp32, y_pred_fp32)
        loss = tf.cond(
            dice_loss < 0.2,
            true_fn=lambda: dice_loss,
            false_fn=lambda: dice_coef_loss_binary
        )

        if mode == tf.estimator.ModeKeys.EVAL:
            return global_step, loss, predictions, labels

        if mode == tf.estimator.ModeKeys.TRAIN:
            return global_step, loss

        raise NotImplementedError(mode)

    def optimizer_function(global_step, loss):
        learning_rate = tf.train.exponential_decay(
            learning_rate=args.learning_rate,
            global_step=global_step,
            decay_steps=args.lr_steps,
            decay_rate=args.lr_decay_rate,
            staircase=True
        )

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=0.9,
            momentum=0.8,
            epsilon=1e-3,
            centered=True
        )

        def map_fn_decay(grad, var):
            return grad + (args.weight_decay * var)

        if args.weight_decay > 0:
            optimizer = ipu.optimizers.MapGradientOptimizer(
                optimizer, map_fn_decay)

        return ipu.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

    def eval_metrics_fn(global_step, loss, predictions, labels):
        return {
            "loss": loss
        }

    if args.pipeline_schedule == "Interleaved":
        pipeline_schedule = ipu.pipelining_ops.PipelineSchedule.Interleaved
    elif args.pipeline_schedule == "Grouped":
        pipeline_schedule = ipu.pipelining_ops.PipelineSchedule.Grouped
    elif args.pipeline_schedule == "Sequential":
        pipeline_schedule = ipu.pipelining_ops.PipelineSchedule.Sequential
    else:
        raise NotImplementedError(args.pipeline_schedule)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # For inference we do not need pipelining
        return pipeline_stage2(*pipeline_stage1(tf.constant(0), inference_features, tf.constant(0)))

    elif mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        # For training and evaluation we pipeline the graph across several IPUs
        global_step_input = tf.cast(
            tf.train.get_global_step(), dtype=tf.float32)
        return ipu.ipu_pipeline_estimator.IPUPipelineEstimatorSpec(
            mode,
            computational_stages=[pipeline_stage1, pipeline_stage2],
            optimizer_function=optimizer_function,
            eval_metrics_fn=eval_metrics_fn,
            inputs=[global_step_input],
            offload_weight_update_variables=(
                args.offload_weight_update_variables == 1),
            pipeline_schedule=pipeline_schedule,
            gradient_accumulation_count=args.gradient_accumulation_batches)


class ThroughputCalcHook(tf.train.SessionRunHook):
    """Estimator hook to calculate mean throughput"""

    def __init__(self, images_per_run, warmup_steps):
        self._images_per_run = images_per_run
        self._step = 0
        self._warmup_steps = warmup_steps
        self._throughput_history = []

    def before_run(self, run_context):
        self._step += 1
        self._time_before_run = time.time()

    def after_run(self, run_context, run_values):
        delta = time.time() - self._time_before_run
        throughput = self._images_per_run / delta
        if self._step > self._warmup_steps:
            self._throughput_history.append(throughput)

    def end(self, session):
        if self._throughput_history:
            average_throughput = float(np.mean(self._throughput_history))
            print(f"\nMean throughput: {average_throughput:.2f} images/sec\n")


def create_estimator(args):
    cfg = IPUConfig()
    cfg.floating_point_behaviour.inv = True
    cfg.floating_point_behaviour.div0 = True
    cfg.floating_point_behaviour.oflo = True
    cfg.floating_point_behaviour.esr = bool(args.stochastic_rounding)
    cfg.floating_point_behaviour.nanoo = True

    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 20000000

    if args.allow_recompute:
        cfg.allow_recompute = True

    num_replicas = args.num_replicas_train
    num_shards = args.num_ipus_in_pipeline_train

    cfg.auto_select_ipus = num_replicas * num_shards

    cfg.device_connection.version = 'ipu' + str(2)
    cfg.device_connection.type = ipu.utils.DeviceConnectionType.ALWAYS

    cfg.convolutions.poplar_options = {'partialsType': 'half' if args.partials_type == 'float16' else 'float'}
    cfg.matmuls.poplar_options = {'partialsType': 'half' if args.partials_type == 'float16' else 'float'}

    iterations_per_loop = (args.batches_per_step *
                           args.gradient_accumulation_batches)

    ipu_run_config = ipu.ipu_run_config.IPURunConfig(
        iterations_per_loop=iterations_per_loop,
        num_replicas=num_replicas,
        num_shards=num_shards,
        ipu_options=cfg,
    )

    config = ipu.ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config,
        log_step_count_steps=args.log_interval,
        save_summary_steps=args.summary_interval,
        model_dir=args.model_dir,
        tf_random_seed=42
    )

    return ipu.ipu_pipeline_estimator.IPUPipelineEstimator(
        config=config,
        model_fn=partial(model_fn, args=args),
        params={},
    )


def data_fn(args, mode, count_only=False):
    # Generate random data
    dtype = np.float16 if args.dtype == 'float16' else np.float32
    bs = args.batch_size_train if mode == tf.estimator.ModeKeys.TRAIN else args.batch_size_infer
    l = args.batches_per_step * bs
    if count_only:
        return l * 10
    s = args.input_size
    c = args.input_channels
    x = np.random.uniform(size=(l, s, s, c)).astype(dtype)
    y = np.random.randint(args.output_classes,
                          size=(l, s * s * 1)).astype(dtype)

    # Build dataset
    if mode == tf.estimator.ModeKeys.PREDICT:
        dataset = tf.data.Dataset.from_tensor_slices(x)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(
        bs, drop_remainder=True).prefetch(l).cache().repeat()
    return dataset


def train(estimator, args):
    """Train a model and save checkpoints to the given `args.model_dir`."""

    # Training progress is logged as INFO, so enable that logging level
    tf.logging.set_verbosity(tf.logging.INFO)

    num_train_examples = data_fn(
        args, mode=tf.estimator.ModeKeys.TRAIN, count_only=True)
    num_items_to_train_on = num_train_examples * args.epochs

    print(f"Items in dataset: {num_train_examples}")

    total_batch_size = args.batch_size_train * args.num_replicas_train
    steps = num_items_to_train_on // total_batch_size
    iterations_per_loop = (args.batches_per_step *
                           args.gradient_accumulation_batches)
    # IPUEstimator requires no remainder; steps must be divisible by iterations_per_loop
    steps += (iterations_per_loop - steps % iterations_per_loop)

    # Set up hooks
    hooks = [ThroughputCalcHook(total_batch_size *
                                iterations_per_loop, args.warmup_steps)]

    with tf.device("cpu"):
        tf.train.get_or_create_global_step()

    if args.profile:
        steps = iterations_per_loop

    t0 = time.time()
    input_fn = partial(data_fn, args=args, mode=tf.estimator.ModeKeys.TRAIN)
    estimator.train(input_fn=input_fn, steps=steps, hooks=hooks)
    t1 = time.time()

    duration_seconds = t1 - t0
    print(f"Took {duration_seconds:.2f} seconds to compile and run")


def evaluate(estimator, args):
    """Evaluate the model by loading weights from the final checkpoint in the
    given `args.model_dir`."""

    num_eval_examples = data_fn(
        args, mode=tf.estimator.ModeKeys.EVAL, count_only=True)
    print(f"Items in dataset: {num_eval_examples}")

    total_batch_size = args.batch_size_infer * args.num_replicas_infer
    steps = num_eval_examples // total_batch_size
    iterations_per_loop = (args.batches_per_step *
                           args.gradient_accumulation_batches)
    # IPUEstimator requires no remainder; steps must be divisible by iterations_per_loop
    steps -= steps % iterations_per_loop
    if steps == 0:
        print(f"Repeating the evaluation dataset to make at least 1 step")
        steps = iterations_per_loop

    if args.profile:
        steps = iterations_per_loop

    print(f"Evaluating on {steps * total_batch_size} examples")

    t0 = time.time()
    input_fn = partial(data_fn, args=args, mode=tf.estimator.ModeKeys.EVAL)
    metrics = estimator.evaluate(input_fn=input_fn, steps=steps)
    t1 = time.time()

    print("Evaluation dice coeficient: {:g}".format(1-metrics["loss"]))
    print("Evaluation loss: {:g}".format(metrics["loss"]))


def inference_test(args):
    # Create data feeds
    ds = data_fn(args, mode=tf.estimator.ModeKeys.PREDICT)
    infeed = ipu.ipu_infeed_queue.IPUInfeedQueue(ds)
    outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    inference_batches_per_step = args.batches_per_step * \
        args.gradient_accumulation_batches

    def test_loop_op(args=None):
        def body(features):
            predictions = model_fn(
                inference_features=features, mode=tf.estimator.ModeKeys.PREDICT, params=[], args=args)
            outfeed_op = outfeed.enqueue(predictions)
            return outfeed_op
        return ipu.loops.repeat(inference_batches_per_step, body, infeed_queue=infeed)

    with ipu.scopes.ipu_scope('/device:IPU:0'):
        compiled = ipu.ipu_compiler.compile(partial(test_loop_op, args=args))

    with tf.Session() as sess:
        # Initialize
        ipu.utils.move_variable_initialization_to_cpu()
        init_g = tf.global_variables_initializer()
        sess.run(infeed.initializer)
        sess.run(init_g)
        outfeed_dequeue_op = outfeed.dequeue()

        print(f"Warming up...")
        sess.run(compiled)
        r = sess.run(outfeed_dequeue_op)

        loop_steps = 10 if not args.profile else 1
        num_items = loop_steps * inference_batches_per_step * \
            args.batch_size_infer * args.num_replicas_infer
        print(f"Inferring on {num_items} items")

        # Receive predictions in a separate thread
        def dequeue_thread_fn():
            counter = 0
            while counter != num_items:
                r = sess.run(outfeed_dequeue_op)
                if r.size:
                    counter += np.product(r.shape[0:2]
                                          if args.num_replicas_infer == 1 else r.shape[0:3])
            print(f"Received results for {counter} items")

        if not args.use_synthetic_data:
            dequeue_thread = Thread(target=dequeue_thread_fn)
            dequeue_thread.start()

        t0 = time.time()
        tp = t0
        for step in range(loop_steps):
            if time.time() - tp > 5:
                # Print step number if 5 seconds have passed since we printing the last time
                print(f'Step {step+1}/{loop_steps}')
                tp = time.time()
            sess.run(compiled)
        t1 = time.time()

        if not args.use_synthetic_data:
            dequeue_thread.join()

    duration_seconds = t1 - t0
    throughput = num_items / duration_seconds
    print(f"Took {duration_seconds:.2f} seconds to run")
    print(f"Processed {num_items} items")
    print(f"Throughput: {throughput:.1f} items/second")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dtype",
        default='float16',
        type=str,
        help="Floating point type: float16 or float32")

    parser.add_argument(
        "--partials-type",
        default='float16',
        type=str,
        help="Matmul partial results floating point type: float16 or float32")

    parser.add_argument(
        "--master-weight-type",
        default='float32',
        type=str,
        help="Precision of master weights: float16 or float32")

    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=1,
        help="Batch size for training")

    parser.add_argument(
        "--batch-size-infer",
        type=int,
        default=4,
        help="Batch size for inference")

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of warm up steps for throughput measuring")

    parser.add_argument(
        "--batches-per-step",
        type=int,
        default=50,
        help="Number of batches per execution loop on IPU")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total number of epochs to train for")

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate used with stochastic gradient descent")

    parser.add_argument(
        "--lr-decay-rate",
        type=float,
        default=0.8,
        help="Learning rate decay rate")

    parser.add_argument(
        "--lr-steps",
        type=int,
        default=500,
        help="Learning rate decay steps")

    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="Input image height and width")

    parser.add_argument(
        "--input-channels",
        type=int,
        default=1,
        help="Number of channels in input images")

    parser.add_argument(
        "--output-classes",
        type=int,
        default=2,
        help="Number of channels in input images")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Interval at which to log progress")

    parser.add_argument(
        "--summary-interval",
        type=int,
        default=1,
        help="Interval at which to write summaries")

    parser.add_argument(
        "--model-dir",
        help="Directory where checkpoints and summaries are stored")

    parser.add_argument(
        "--profile",
        action='store_true',
        help="Generate compilation and execution report"
    )

    parser.add_argument(
        "--profile-directory",
        default="./profiling_report",
        help="Path to store the generated compilation and execution report"
    )

    parser.add_argument(
        "--training",
        action='store_true',
        help="Run training"
    )

    parser.add_argument(
        "--evaluation",
        action='store_true',
        help="Run evaluation"
    )

    parser.add_argument(
        "--inference",
        action='store_true',
        help="Run inference"
    )

    parser.add_argument(
        "--use-synthetic-data",
        action='store_true',
        help="Test performance without the overhead of data transfer"
    )

    parser.add_argument(
        "--use-random-data",
        action='store_true',
        help="Test performance with random data (no train/test datasets needed)"
    )

    parser.add_argument(
        "--null-data-feed",
        action='store_true',
        help="Ignore tf.dataset feed and transfer random data to IPU"
    )

    parser.add_argument(
        "--num-replicas-train",
        type=int,
        default=2,
        help="Number of graph replicas for training and evaluation")

    parser.add_argument(
        "--num-ipus-in-pipeline-train",
        type=int,
        default=2,
        help="Number of IPUs in pipeline for training (do not change this)")

    parser.add_argument(
        "--num-replicas-infer",
        type=int,
        default=4,
        help="Number of graph replicas for inference")

    parser.add_argument(
        "--num-ipus-in-pipeline-infer",
        type=int,
        default=1,
        help="Number of IPUs in pipeline for inference (do not change this)")

    parser.add_argument(
        "--pipeline-schedule",
        default="Grouped",
        help="Optimizer: Grouped, Interleaved, Sequential")

    parser.add_argument(
        "--gradient-accumulation-batches",
        type=int,
        default=32,
        help="Number of IPU and graph replicas")

    parser.add_argument(
        "--offload-weight-update-variables",
        type=int,
        default=0,
        help="Store weight update variables in the remote memory: 1=True, 0=False")

    parser.add_argument(
        "--stochastic-rounding",
        type=int,
        default=1,
        help="Enable or disable stochastic rounding: 1=True, 0=False")

    parser.add_argument(
        "--allow-recompute",
        type=int,
        default=0,
        help="Allow recomputation: 1=True, 0=False")

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay rate")

    return parser.parse_args()


if __name__ == "__main__":
    os.environ["GCDA_MONITOR"] = "1"

    # Assert TensorFlow version compatibility
    assert version.parse(tf.__version__) > version.parse("1.15") and \
        version.parse(tf.__version__) < version.parse("2"), \
        f"TensorFlow version is {tf.__version__}, but version >=1.15 and <2 is required"

    # Parse args, check for correctness, and apply to OS variables
    args = parse_args()

    assert args.output_classes == 2, f"Number of output classes must be 2 (given {args.output_classes})"
    assert args.num_ipus_in_pipeline_train == 2, "num_ipus_in_pipeline_train must be equal to 2"
    assert args.num_ipus_in_pipeline_infer == 1, "num_ipus_in_pipeline_infer must be equal to 1"

    if args.training + args.evaluation + args.inference == 0:
        args.training, args.evaluation, args.inference = True, False, True

    if args.profile and (args.training + args.evaluation + args.inference != 1):
        print("\n\nDid you mean to profile only one of training, evaluation, or inference?"
              "\nConsider using 1 out of these 3 command line arguments: --training --evaluation --inference\n")
        exit(0)

    if args.num_replicas_train * args.num_ipus_in_pipeline_train != \
       args.num_replicas_infer * args.num_ipus_in_pipeline_infer and \
       ((args.training and args.evaluation) or (args.training and args.inference)):
        raise RuntimeError(
            f"Number of IPUs used for training and evaluation/inference must be equal "
            f"(given {args.num_replicas_train * args.num_ipus_in_pipeline_train} for training and "
            f"{args.num_replicas_infer * args.num_ipus_in_pipeline_infer} for inference).")

    if "TF_POPLAR_FLAGS" not in os.environ:
        os.environ["TF_POPLAR_FLAGS"] = ""
    if args.use_synthetic_data:
        os.environ["TF_POPLAR_FLAGS"] += " --use_synthetic_data --synthetic_data_initializer=random"
    if args.null_data_feed:
        os.environ["TF_POPLAR_FLAGS"] += " --null_data_feed"

    if args.profile:
        os.environ["POPLAR_ENGINE_OPTIONS"] = f'{{"autoReport.all":"true", "autoReport.directory":"{args.profile_directory}", "debug.allowOutOfMemory":"true"}}'

    # Print the about message
    print("\n\n\nUNet Industrial example\nArguments:")
    print(args)
    print("\n\n")

    # Make estimator
    estimator = create_estimator(args)

    if args.training:
        print("\nTraining...")
        train(estimator, args)

    if args.evaluation:
        print("\nEvaluating...")
        evaluate(estimator, args)

    if not (args.training or args.evaluation):
        # Configure IPU system for inference only
        # (no need to do this if an Estimator was already initialized)
        cfg = IPUConfig()
        if args.allow_recompute:
            cfg.allow_recompute = True
        cfg.auto_select_ipus = (args.num_replicas_infer * args.num_ipus_in_pipeline_infer)
        cfg.device_connection.version = 'ipu' + str(2)
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ALWAYS
        cfg.convolutions.poplar_options = {'partialsType': 'half' if args.artials_type == 'float16' else 'float'}
        cfg.matmuls.poplar_options = {'partialsType': 'half' if args.partials_type == 'float16' else 'float'}
        cfg.configure_ipu_system()

    if args.inference:
        print("\nTesting inference...")
        inference_test(args)

    print("\nCompleted")
