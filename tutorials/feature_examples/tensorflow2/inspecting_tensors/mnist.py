# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

""" The main script for this example.

Trains a choice of simple fully connected models on the MNIST
numeral data set and shows how tensors (containing activations and gradients)
can be returned to the host via outfeeds for inspection.
"""

import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

from outfeed_callback import OutfeedCallback
from outfeed_optimizer import OutfeedOptimizer, OutfeedOptimizerMode
import outfeed_layers
from outfeed_wrapper import MaybeOutfeedQueue

if tf.__version__[0] != "2":
    raise ImportError("TensorFlow 2 is required for this example")


def create_dataset():
    """Create the Dataset containing input data and labels."""
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(32, drop_remainder=True)
    )
    train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    return train_ds.repeat()


def create_sequential_model(multi_activations_outfeed_queue, gradient_accumulation_steps_per_replica):
    """Create the model using the Keras Sequential class.

    Outfeed the activations for multiple layers.
    """
    model = keras.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu", name="Dense_128"),
            outfeed_layers.MaybeOutfeed(
                multi_activations_outfeed_queue,
                final_outfeed=False,
                name="Dense_128_acts",
            ),
            keras.layers.Dense(10, activation="softmax", name="Dense_10"),
            outfeed_layers.MaybeOutfeed(
                multi_activations_outfeed_queue,
                final_outfeed=True,
                name="Dense_10_acts",
            ),
        ]
    )
    model.set_gradient_accumulation_options(
        gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica
    )
    return model


def create_model(activations_outfeed_queue, gradient_accumulation_steps_per_replica):
    """Create the model using the Keras Model class.

    Outfeed the activations for a single layer.
    """
    input_layer = keras.layers.Input(shape=(28, 28, 1), dtype=tf.float32, batch_size=32)
    x = keras.layers.Flatten()(input_layer)
    x = keras.layers.Dense(128, activation="relu", name="Dense_128")(x)
    # Outfeed the activations for a single layer:
    x = outfeed_layers.Outfeed(activations_outfeed_queue, name="Dense_128_acts")(x)
    x = keras.layers.Dense(10, activation="softmax", name="Dense_10")(x)
    model = keras.Model(input_layer, x)
    model.set_gradient_accumulation_options(
        gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica
    )
    return model


def create_pipeline_model(multi_activations_outfeed_queue, gradient_accumulation_steps_per_replica):
    """Create the model using the Keras Model class.

    Outfeed the activations for multiple layers in the second stage.
    """
    input_layer = keras.layers.Input(shape=(28, 28, 1), dtype=tf.float32, batch_size=32)

    with keras.ipu.PipelineStage(0):
        x = keras.layers.Flatten()(input_layer)
        x = keras.layers.Dense(256, activation="relu", name="Dense_256")(x)

    with keras.ipu.PipelineStage(1):
        x = keras.layers.Dense(128, activation="relu", name="Dense_128")(x)
        x = outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue, final_outfeed=False, name="Dense_128_acts")(x)
        x = keras.layers.Dense(10, activation="softmax", name="Dense_10")(x)
        x = outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue, final_outfeed=True, name="Dense_10_acts")(x)
    model = keras.Model(input_layer, x)
    model.set_pipelining_options(gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica)
    return model


def create_pipeline_sequential_model(multi_activations_outfeed_queue, gradient_accumulation_steps_per_replica):
    """Create the model using the Keras Sequential class.
    Pipeline the model by assigning layers to stages through
    `set_pipeline_stage_assignment`.
    Outfeed the activations for multiple layers in the second stage.
    """
    model = keras.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu", name="Dense_256"),
            keras.layers.Dense(128, activation="relu", name="Dense_128"),
            outfeed_layers.MaybeOutfeed(
                multi_activations_outfeed_queue,
                final_outfeed=False,
                name="Dense_128_acts",
            ),
            keras.layers.Dense(10, activation="softmax", name="Dense_10"),
            outfeed_layers.MaybeOutfeed(
                multi_activations_outfeed_queue,
                final_outfeed=True,
                name="Dense_10_acts",
            ),
        ]
    )
    model.set_pipelining_options(gradient_accumulation_steps_per_replica=gradient_accumulation_steps_per_replica)
    model.set_pipeline_stage_assignment([0, 0, 1, 1, 1, 1])
    return model


def main():

    model_type_options = ["Model", "Sequential"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default="Sequential",
        choices=model_type_options,
        help="The model type to use",
    )
    parser.add_argument("--no-pipelining", action="store_true", help="Do not use pipelining")
    parser.add_argument(
        "--outfeed-pre-accumulated-gradients",
        action="store_true",
        help="Outfeed the pre-accumulated rather than accumulated gradients."
        " Only makes a difference when using gradient accumulation"
        " (which is the case when pipelining).",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=2000,
        help="Number of steps to run per epoch.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument(
        "--gradients-filters",
        nargs="+",
        type=str,
        default=["Dense_128"],
        help="Space separated strings for determining which gradients"
        " to add to the dict that is enqueued on the outfeed queue."
        " Pass '--gradients-filters none' to disable filtering.",
    )
    parser.add_argument(
        "--activations-filters",
        nargs="+",
        type=str,
        default=["none"],
        help="Space separated strings for determining which activations"
        " in the second PipelineStage"
        " to add to the dict that is enqueued on the outfeed queue."
        " Pass '--activations-filters none' to disable filtering."
        " (Only applicable for the Pipeline models.)",
    )
    parser.add_argument(
        "--use-gradient-accumulation",
        action="store_true",
        help="Use gradient accumulation (always true when pipelining)",
    )

    args = parser.parse_args()

    if not args.no_pipelining:
        num_ipus = 2
        args.use_gradient_accumulation = True
    else:
        num_ipus = 1

    print(args)

    gradient_accumulation_steps_per_replica = 4

    if args.outfeed_pre_accumulated_gradients:
        outfeed_optimizer_mode = OutfeedOptimizerMode.AFTER_COMPUTE
    else:
        outfeed_optimizer_mode = OutfeedOptimizerMode.BEFORE_APPLY

    # Configure the IPU system
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = num_ipus
    cfg.configure_ipu_system()

    # Create an IPU distribution strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()

    def process_filters(filters_input):
        if len(filters_input) == 1 and filters_input[0].lower() == "none":
            return None
        return filters_input

    with strategy.scope():
        # Create the outfeed queue for selected gradients
        optimizer_outfeed_queue = MaybeOutfeedQueue(filters=process_filters(args.gradients_filters))
        # Remove the filters to get the gradients for all layers
        # or pass different strings to the argument to select other layer(s)

        # Create the outfeed queues for the custom layers
        activations_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        multi_activations_outfeed_queue = MaybeOutfeedQueue(filters=process_filters(args.activations_filters))

        # Create a callback for the gradients
        gradients_outfeed_callback = OutfeedCallback(optimizer_outfeed_queue, name="Gradients callback")

        # Create callbacks for the activations in the custom layers
        layer_outfeed_callback = OutfeedCallback(activations_outfeed_queue, name="Single layer activations callback")

        multi_layer_outfeed_callback = OutfeedCallback(
            multi_activations_outfeed_queue, name="Multi-layer activations callback"
        )

        callbacks = [gradients_outfeed_callback]

        # Create an instance of the model.
        if not args.no_pipelining:
            if args.model_type == "Model":
                model = create_pipeline_model(
                    multi_activations_outfeed_queue,
                    gradient_accumulation_steps_per_replica,
                )
                callbacks += [multi_layer_outfeed_callback]
            elif args.model_type == "Sequential":
                model = create_pipeline_sequential_model(
                    multi_activations_outfeed_queue,
                    gradient_accumulation_steps_per_replica,
                )
                callbacks += [multi_layer_outfeed_callback]
        else:
            if args.use_gradient_accumulation:
                gradient_accumulation_steps_per_replica = gradient_accumulation_steps_per_replica
            else:
                gradient_accumulation_steps_per_replica = 1
            if args.model_type == "Sequential":
                model = create_sequential_model(
                    multi_activations_outfeed_queue,
                    gradient_accumulation_steps_per_replica,
                )
                callbacks += [multi_layer_outfeed_callback]
            elif args.model_type == "Model":
                model = create_model(activations_outfeed_queue, gradient_accumulation_steps_per_replica)
                callbacks += [layer_outfeed_callback]

        # Get the training dataset.
        dataset = create_dataset()

        # Build the graph, passing an OutfeedOptimizer to enqueue selected gradients
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=OutfeedOptimizer(
                keras.optimizers.SGD(),
                optimizer_outfeed_queue,
                outfeed_optimizer_mode=outfeed_optimizer_mode,
                model=model,
            ),
            steps_per_execution=args.steps_per_epoch,
        )

        # Train the model, passing the callbacks to see the gradients and activations stats
        model.fit(
            dataset,
            callbacks=callbacks,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
