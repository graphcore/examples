# Copyright 2020 Graphcore Ltd.
import argparse
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.python import ipu
from tensorflow.python.ipu.ipu_session_run_hooks import IPULoggingTensorHook

NUM_CLASSES = 10


def model_fn(features, labels, mode, params):
    """A simple CNN based on https://keras.io/examples/cifar10_cnn/"""

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))

    logits = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(input=logits, axis=-1)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions),
        }
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        # In a single device loop, the IPU executes n batches without returning
        # to the host in between. Each batch flows through the model_fn
        # defined here, causing all tensors in this graph to have values.
        # Any of these values can be logged to the screen using an
        # IPULoggingTensorHook and its 'log' method.
        loss_logging_hook = IPULoggingTensorHook(
            # As the IPUEstimator executes device loops, values will accrue in
            # the hook's outfeed. To only ever store and print the last accrued
            # value, use LoggingMode.LAST. To store and print all values accrued
            # between logs, use LoggingMode.ALL.
            # Here, we average the ALL values.
            logging_mode=IPULoggingTensorHook.LoggingMode.ALL,
            formatter=lambda dct: ' '.join(
                {f" {name} = {np.mean(val):.3f}" for name, val in dct.items()}
            ),
            feed_name="loss_logging_hook",
            # The frequency of logging does not have to align with the frequency
            # of device loops. However note that, while outfeeds can
            # theoretically be concurrently enqueued and dequeued, Estimators
            # sequentially interleave calls to device loops and hooks, therefore
            # the logging frequency is bounded below by the device loop
            # frequency.
            every_n_secs=params["log_interval"],
        )
        log_loss = loss_logging_hook.log(loss)
        optimizer = tf.train.GradientDescentOptimizer(params["learning_rate"])
        train_op = optimizer.minimize(loss=loss)
        # The logging operation must be forced to be part of the graph's
        # control flow, similar to 'tf.print'
        train_op = tf.group([train_op, log_loss])
        # Add the logging hook to the spec's training_hooks
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[loss_logging_hook])
    else:
        raise NotImplementedError(mode)


class ProfilerHook(tf.train.SessionRunHook):
    """Estimator hook to generate and write a Poplar report to write_dir"""
    def __init__(self, write_dir, name=''):
        self._write_dir = write_dir
        self._name = name

    def begin(self):
        from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
        self._report_op = gen_ipu_ops.ipu_event_trace()

    def end(self, session):
        import os
        raw_report = session.run(self._report_op)
        write_file = os.path.join(self._write_dir, f'{self._name}_report.txt')
        with open(write_file, 'w', encoding="utf-8") as f:
            f.write(ipu.utils.extract_all_strings_from_event_trace(raw_report))
        print(f"Wrote profiling report to {write_file}")


def create_ipu_estimator(args):
    ipu_options = ipu.utils.create_ipu_config(
        profiling=args.profile,
        use_poplar_text_report=args.profile,
        profile_execution=args.profile
    )

    ipu.utils.auto_select_ipus(ipu_options, num_ipus=1)

    ipu_run_config = ipu.ipu_run_config.IPURunConfig(
        iterations_per_loop=args.batches_per_step,
        ipu_options=ipu_options,
    )

    config = ipu.ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config,
        # Turn off default Estimator logging
        log_step_count_steps=None,
        save_summary_steps=args.summary_interval,
        model_dir=args.model_dir,
    )

    return ipu.ipu_estimator.IPUEstimator(
        config=config,
        model_fn=model_fn,
        params={"learning_rate": args.learning_rate,
                "log_interval": args.log_interval},
    )


def train(ipu_estimator, args, x_train, y_train):
    """
    Train a model on IPU and save checkpoints to the given `args.model_dir`.
    """
    def input_fn():
        # If using Dataset.from_tensor_slices, the data will be embedded
        # into the graph as constants, which makes the training graph very
        # large and impractical. So use Dataset.from_generator here instead.

        def generator(): return zip(x_train, y_train)
        types = (x_train.dtype, y_train.dtype)
        shapes = (x_train.shape[1:], y_train.shape[1:])

        dataset = tf.data.Dataset.from_generator(generator, types, shapes)
        dataset = dataset.prefetch(len(x_train)).cache()
        dataset = dataset.repeat()
        dataset = dataset.shuffle(len(x_train))
        dataset = dataset.batch(args.batch_size, drop_remainder=True)

        return dataset

    # Training progress is logged as INFO, so enable that logging level
    tf.logging.set_verbosity(tf.logging.INFO)

    # To evaluate N epochs each of n data points, with batch size BS
    # do Nn / BS steps.
    num_train_examples = int(args.epochs * len(x_train))
    steps = num_train_examples // args.batch_size
    # IPUEstimator requires no remainder; batches_per_step must divide steps
    steps += (args.batches_per_step - steps % args.batches_per_step)

    # Set up profiling hook
    hooks = []
    if args.profile:
        hooks.append(ProfilerHook(ipu_estimator.model_dir, name='train'))

    t0 = time.time()
    ipu_estimator.train(input_fn=input_fn,
                        steps=steps,
                        hooks=hooks)
    t1 = time.time()

    duration_seconds = t1 - t0
    print(f"Took {duration_seconds:.2f} seconds to compile and run")


def test(ipu_estimator, args, x_test, y_test):
    """
    Test the model on IPU by loading weights from the final checkpoint in the
    given `args.model_dir`.
    """

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset = dataset.prefetch(len(x_test)).cache()
        dataset = dataset.batch(args.batch_size, drop_remainder=True)
        return dataset

    num_test_examples = len(x_test)
    steps = num_test_examples // args.batch_size
    # IPUEstimator requires no remainder; batches_per_step must divide steps
    steps -= steps % args.batches_per_step
    print(f"Evaluating on {steps * args.batch_size} examples")

    # Set up profiling hook
    hooks = []
    if args.profile:
        hooks.append(ProfilerHook(ipu_estimator.model_dir, name='eval'))

    t0 = time.time()
    metrics = ipu_estimator.evaluate(input_fn=input_fn,
                                     steps=steps,
                                     hooks=hooks)
    t1 = time.time()

    test_loss = metrics["loss"]
    test_accuracy = metrics["accuracy"]
    duration_seconds = t1 - t0
    print("Test loss: {:g}".format(test_loss))
    print("Test accuracy: {:.2f}%".format(100 * test_accuracy))
    print(f"Took {duration_seconds:.2f} seconds to compile and run")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size.")

    parser.add_argument(
        "--batches-per-step",
        type=int,
        default=100,
        help="The number of batches per execution loop on IPU.")

    parser.add_argument(
        "--epochs",
        type=float,
        default=100,
        help="Total number of epochs to train for.")

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="The learning rate used with stochastic gradient descent.")

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and test using latest checkpoint from model_dir.")

    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run training only and skip the testing."
    )

    parser.add_argument(
        "--log-interval",
        type=float,
        default=3,
        help="Interval at which to log progress (seconds)")

    parser.add_argument(
        "--summary-interval",
        type=int,
        default=1,
        help="Interval at which to write summaries.")

    parser.add_argument(
        "--model-dir",
        help="Directory where checkpoints and summaries are stored.")

    parser.add_argument(
        "--profile",
        action='store_true',
        help="Generate compilation and execution report, written to model-dir"
    )

    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Run the model with synthetic data."
    )

    return parser.parse_args()


def synthetic_dataset(number_of_samples):
    """Returns cifar10 shaped dataset filled with random uint8."""
    cifar10_shape = (number_of_samples, 32, 32, 3)
    np.random.seed(0)
    x_data = np.random.randint(1, size=cifar10_shape, dtype="uint8")
    y_data = np.random.randint(1, size=(number_of_samples,), dtype="uint8")
    y_data = np.reshape(y_data, (len(y_data), 1))
    return (x_data, y_data)


if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Load data
    if args.synthetic_data:
        train_data = test_data = synthetic_dataset(50000)
    else:
        train_data, test_data = cifar10.load_data()

    # Make estimator
    ipu_estimator = create_ipu_estimator(args)

    # Train and test
    def normalise(x, y):
        return x.astype("float32") / 255.0, y.astype("int32")

    if not args.test_only:
        print("Training...")
        x_train, y_train = normalise(*train_data)
        train(ipu_estimator, args, x_train, y_train)

    if not args.train_only:
        print("Testing...")
        x_test, y_test = normalise(*test_data)
        test(ipu_estimator, args, x_test, y_test)
