# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import os

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.python import ipu
from custom_ops import sharded
from functools import partial


tf.disable_eager_execution()
tf.disable_v2_behavior()


def parse_args():
    # Handle command line arguments
    pipeline_schedule_options = [p.name for p in ipu.pipelining_ops.PipelineSchedule]

    parser = argparse.ArgumentParser("Concurrent pipeline example.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="The (micro-)batch size.")
    parser.add_argument("--repeat-count", type=int, default=100,
                        help="The number of times the pipeline will be executed for each session.run step.")
    parser.add_argument("--epochs", type=float, default=30,
                        help="Total number of epochs to train over.")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="The learning rate used with stochastic gradient descent.")
    parser.add_argument("--gradient-accumulation-count", type=int, default=84,
                        help="The number of batches to accumulate gradients over before applying a weight update.")
    parser.add_argument("--pipeline-schedule", type=str, default="Grouped",
                        choices=pipeline_schedule_options,
                        help="Pipelining schedule. In the 'Grouped' schedule the forward passes"
                        " are grouped together, and the backward passes are grouped together. "
                        "With 'Interleaved' the forward and backward passes are interleaved. "
                        "'Sequential' mimics a non-pipelined execution.")
    parser.add_argument("--synthetic-data", action="store_true",
                        help="Use synthetic data instead of real images.")
    parser.add_argument("--run-single-step", action="store_true",
                        help="Shorten the run for profiling: runs for a single step.")
    parser.add_argument("--pipeline-mode", type=str, required=True, choices=["basic", "concurrent"],
                        help="Choose between the two types of pipeline: basic (where one IPU works on each stage)"
                             " or concurrent (where stages can be tensor parallel across both IPUs).")
    args = parser.parse_args()
    return args


def create_dataset(args):
    # Prepare a tf dataset with mnist data
    train_data, _ = mnist.load_data()

    def normalise(x, y):
        return x.astype("float32") / 255.0, y.astype("int32")

    x_train, y_train = normalise(*train_data)

    def generator():
        return zip(x_train, y_train)

    types = (x_train.dtype, y_train.dtype)
    shapes = (x_train.shape[1:], y_train.shape[1:])

    n_examples = len(x_train)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    # Caching and prefetching are important to prevent the host data
    # feed from being the bottleneck for throughput
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return n_examples, dataset


# The following is a schematic representation of the basic pipeline
# model which splits sequential layers across two IPUs:
# -------------------------- Basic Pipeline -----------------------------
# IPU0: inputs -> MLP \
# IPU1:                |-> Classifier -- SoftmaxCE -- Loss
#
# However, when the concurrent pipeline is used (default) then this changes
# so that the final matrix multiply in the classifier layer and the following
# softmax are executed tensor parallel in a concurrent pipeline stage:
# ---------------------------- Concurrent Pipeline --------------------------------
# IPU0: inputs -> MLP \ -> Classifier(top rows)    -- SoftmaxCE(top)     |-> Combined Loss
# IPU1:                |-> Classifier(bottom rows) -- SoftmaxCE(bottom) /


def glorot_limit(shape):
    return np.sqrt(6.0 / (shape[0] + shape[1]))


def get_init_dense_weights(rng_seed, shape, dtype):
    rng = np.random.default_rng(seed=rng_seed)
    limit = glorot_limit(shape)
    weight_values = rng.uniform(-limit, limit, size=shape)
    return tf.constant(weight_values, dtype=dtype)


class ClosureInitializer(tf.keras.initializers.Initializer):
    """
    Returns initial tensor from call to closed function, ignoring
    shape and dtype args passed into __call__ in favour of any
    bound into the closure.
    """
    def __init__(self, closure):
        self.func = closure

    def __call__(self, shape, dtype, **kwargs):
        return self.func()


def custom_dense(hidden_size, namescope, input, kernel_initializer):
    matmul_opts = {
      "availableMemoryProportion": "0.8",
      "partialsType": "half"
    }
    with tf.variable_scope(namescope, reuse=tf.AUTO_REUSE, use_resource=True):
        weights = tf.get_variable("weights", dtype=input.dtype, trainable=True, initializer=kernel_initializer)
        bias = tf.get_variable("bias", shape = [hidden_size], dtype=input.dtype, trainable=True, initializer=tf.zeros_initializer())
        prod = sharded.matmul(input, weights, matmul_opts, name="result")
        return tf.add(prod, bias)


def mlp_stage(lr, images, labels):
    # Stage 1 of the pipeline. Will be placed on the first IPU
    partial = layers.Flatten()(images)
    partial = layers.Dense(2560, activation=tf.nn.relu, name='stage1_dense1')(partial)
    partial = layers.Dense(1280, activation=tf.nn.relu, name='stage1_dense2')(partial)
    return lr, partial, labels


def classifier_stage(lr, input, labels):
    # Stage 2 of the pipeline. Will be placed on the second IPU
    hidden_size = 10
    weights_shape = [input.shape.as_list()[1], hidden_size]
    rng_seed = 101
    init_func = partial(get_init_dense_weights, rng_seed, weights_shape, tf.float32)

    logits = layers.Dense(hidden_size, name='stage2_dense',
                          kernel_initializer=ClosureInitializer(init_func))(input)
    return lr, logits, labels


def sharded_classifier_and_loss_stage(lr, input, labels):
    opts = {
        "availableMemoryProportion": "0.3",
        "partialsType": "half"
    }
    hidden_size = 10
    weight_shape = [input.shape.as_list()[1], hidden_size]
    weights = tf.get_variable('weights', shape=weight_shape, trainable=True, initializer=tf.glorot_uniform_initializer)
    inputSharded = sharded.to_all(input, 2)
    expanded_labels = tf.expand_dims(labels, 1)
    labelsSharded = sharded.to_all(expanded_labels, 2)
    output = sharded.matmul(inputSharded, weights, opts, name="custom_matmul")
    ce = sharded.log_softmax_cross_entropy(output, labelsSharded)
    loss = tf.reduce_mean(ce, name="final_loss")
    return lr, loss


def loss_stage(lr, logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return lr, loss


def optimizer_function(lr, loss):
    # Optimizer function used by the pipeline to automatically set up
    # the gradient accumulation and weight update steps
    optimizer = tf.train.GradientDescentOptimizer(lr)
    return ipu.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)


def make_basic_pipeline():
    stages = [mlp_stage, classifier_stage, loss_stage]
    devices = [0, ipu.pipelining_ops._ALL_DEVICES, 1]
    return stages, devices


def make_concurrent_pipeline():
    stages = [mlp_stage, sharded_classifier_and_loss_stage]
    devices = [1, ipu.pipelining_ops._ALL_DEVICES]
    return stages, devices


def model(lr):
    if args.pipeline_mode == "basic":
        pipe_stages, pipe_devices = make_basic_pipeline()
    else:
        pipe_stages, pipe_devices = make_concurrent_pipeline()

    # Defines a pipelined model which is split accross two stages:
    sched = next(p for p in ipu.pipelining_ops.PipelineSchedule if args.pipeline_schedule == p.name)
    with tf.variable_scope("FCModel", use_resource=True):
        pipeline_op = ipu.pipelining_ops.pipeline(
            computational_stages=pipe_stages,
            device_mapping=pipe_devices,
            gradient_accumulation_count=args.gradient_accumulation_count,
            repeat_count=args.repeat_count,
            inputs=[lr],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            pipeline_schedule=next(p for p in ipu.pipelining_ops.PipelineSchedule
                                   if args.pipeline_schedule == p.name),
            outfeed_loss=True,
            name="Pipeline")
    return pipeline_op


if __name__ == "__main__":
    args = parse_args()

    if args.synthetic_data:
        if "TF_POPLAR_FLAGS" in os.environ:
            os.environ["TF_POPLAR_FLAGS"] += " --use_synthetic_data"
        else:
            os.environ["TF_POPLAR_FLAGS"] = "--use_synthetic_data"

    n_examples, dataset = create_dataset(args)

    # Create the data queues from/to IPU
    infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

    # With batch size BS, gradient accumulation count GAC and repeat count RPT,
    # at every step n = (BS * GAC * RPT) examples are used.
    # So in order to evaluate at least N total examples, do ceil(N / n) steps
    num_train_examples = int(args.epochs * n_examples)
    examples_per_step = args.batch_size * args.gradient_accumulation_count * args.repeat_count
    steps = ((num_train_examples - 1) // examples_per_step) + 1

    if args.run_single_step:
        steps = 1

    with tf.device('cpu'):
        lr = tf.placeholder(np.float32, [])

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        compiled_model = ipu.ipu_compiler.compile(model, inputs=[lr])

    outfeed_op = outfeed_queue.dequeue()

    ipu.utils.move_variable_initialization_to_cpu()
    init_op = tf.global_variables_initializer()

    # Configure the IPU.
    # With pipelining, IPU-level profiling is needed to correctly visualise the execution trace.
    # For pipelined models either SNAKE or HOOF IPU selection orders are advised;
    # the latter works best when the first and last stage are on the same IPU.
    # For more information, see the API section of the Targeting the IPU from TensorFlow document:
    # https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/tensorflow/api.html#tensorflow.python.ipu.config.SelectionOrder
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.selection_order = ipu.config.SelectionOrder.SNAKE
    cfg.configure_ipu_system()

    with tf.Session() as sess:
        # Initialize
        sess.run(init_op)
        sess.run(infeed_queue.initializer)
        # Run
        for step in range(steps):
            sess.run(compiled_model, {lr: args.learning_rate})

            # Read the outfeed for the training losses
            losses = sess.run(outfeed_op)
            epoch = float(examples_per_step * step / n_examples)
            print("Epoch {:.1f}, Mean loss: {:.3f}".format(
                epoch, np.mean(losses)))
