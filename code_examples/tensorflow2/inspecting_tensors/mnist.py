# Copyright 2020 Graphcore Ltd. All rights reserved.

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

if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")



def create_dataset():
    """ Create the Dataset containing input data and labels.
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
    train_ds = train_ds.map(
        lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    return train_ds.repeat()


def create_sequential_model(activations_outfeed_queue, gradient_accumulation_count):
    """ Create the model using the IPU-specific Sequential class.

        Outfeed the activations for a single layer.
    """
    return ipu.keras.Sequential(
        [keras.layers.Flatten(),
         keras.layers.Dense(128, activation='relu', name="Dense_128"),
         outfeed_layers.Outfeed(activations_outfeed_queue, name="Dense_128_acts"),
         keras.layers.Dense(10, activation='softmax', name="Dense_10")],
        gradient_accumulation_count=gradient_accumulation_count)


def create_model(activations_outfeed_queue, gradient_accumulation_count):
    """ Create the model using the IPU-specific Model class.

        Outfeed the activations for a single layer.
    """
    input_layer = keras.layers.Input(shape=(28, 28, 1), dtype=tf.float32, batch_size=32)
    x = keras.layers.Flatten()(input_layer)
    x = keras.layers.Dense(128, activation='relu', name="Dense_128")(x)
    # Outfeed the activations for a single layer:
    x = outfeed_layers.Outfeed(activations_outfeed_queue, name="Dense_128_acts")(x)
    x = keras.layers.Dense(10, activation='softmax',  name="Dense_10")(x)
    return ipu.keras.Model(input_layer, x, gradient_accumulation_count=gradient_accumulation_count)


def create_pipeline_model(multi_activations_outfeed_queue, gradient_accumulation_count):
    """ Create the model using the IPU-specific PipelineModel class.

        Outfeed the activations for multiple layers in the second stage.
    """
    input_layer = keras.layers.Input(shape=(28, 28, 1), dtype=tf.float32, batch_size=32)

    with ipu.keras.PipelineStage(0):
        x = keras.layers.Flatten()(input_layer)
        x = keras.layers.Dense(256, activation='relu', name="Dense_256")(x)

    with ipu.keras.PipelineStage(1):
        x = keras.layers.Dense(128, activation='relu', name="Dense_128")(x)
        x = outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                        final_outfeed=False, name="Dense_128_acts")(x)
        x = keras.layers.Dense(10, activation='softmax',  name="Dense_10")(x)
        x = outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                        final_outfeed=True, name="Dense_10_acts")(x)

    return ipu.keras.PipelineModel(input_layer, x,
                                   gradient_accumulation_count=gradient_accumulation_count)


def create_pipeline_sequential_model(multi_activations_outfeed_queue,
                                     gradient_accumulation_count):
    """ Create the model using the IPU-specific PipelineSequential class.

        Outfeed the activations for multiple layers in the second stage.
    """
    return ipu.keras.PipelineSequential(
        [[keras.layers.Flatten(),
          keras.layers.Dense(256, activation='relu', name="Dense_256")],
         [keras.layers.Dense(128, activation='relu', name="Dense_128"),
          outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                      final_outfeed=False, name="Dense_128_acts"),
          keras.layers.Dense(10, activation='softmax', name="Dense_10"),
          outfeed_layers.MaybeOutfeed(multi_activations_outfeed_queue,
                                      final_outfeed=True, name="Dense_10_acts")]],
        gradient_accumulation_count=gradient_accumulation_count)


def main():

    model_type_options = ["Model", "Sequential", "PipelineModel", "PipelineSequential"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default="PipelineSequential",
                        choices=model_type_options,
                        help="The model type to use")
    parser.add_argument('--outfeed-pre-accumulated-gradients', action='store_true',
                        help="Outfeed the pre-accumulated rather than accumulated gradients."
                             " Only makes a difference when using gradient accumulation"
                             " (which is the case when pipelining).")
    parser.add_argument('--steps-per-epoch', type=int, default=2000,
                        help="Number of steps to run per epoch.")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument('--gradients-filters', nargs='+', type=str, default=['Dense_128'],
                        help="Space separated strings for determining which gradients"
                             " to add to the dict that is enqueued on the outfeed queue."
                             " Pass '--gradients-filters none' to disable filtering.")
    parser.add_argument('--activations-filters', nargs='+', type=str, default=['none'],
                        help="Space separated strings for determining which activations"
                             " in the second PipelineStage"
                             " to add to the dict that is enqueued on the outfeed queue."
                             " Pass '--activations-filters none' to disable filtering."
                             " (Only applicable for the Pipeline models.)")
    parser.add_argument('--gradient-accumulation', action='store_true',
                        help="Use gradient accumulation (only valid if not pipelining)")

    args = parser.parse_args()

    print(args)

    GRADIENT_ACCUMULATION_COUNT = 4

    if "Pipeline" in args.model_type:
        num_ipus = 2
    else:
        num_ipus = 1

    if args.outfeed_pre_accumulated_gradients:
        outfeed_optimizer_mode = OutfeedOptimizerMode.AFTER_COMPUTE
    else:
        outfeed_optimizer_mode = OutfeedOptimizerMode.BEFORE_APPLY

    # Configure the IPU system
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, num_ipus)
    ipu.utils.configure_ipu_system(cfg)

    # Create an IPU distribution strategy.
    strategy = ipu.ipu_strategy.IPUStrategy()

    def process_filters(filters_input):
        if len(filters_input) == 1 and filters_input[0].lower() == "none":
            return None
        return filters_input

    with strategy.scope():
        # Create the outfeed queue for selected gradients
        optimizer_outfeed_queue = MaybeOutfeedQueue(
                "optimizer_outfeed", filters=process_filters(args.gradients_filters))
        # Remove the filters to get the gradients for all layers
        # or pass different strings to the argument to select other layer(s)

        # Create the outfeed queues for the custom layers
        activations_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue("activations_outfeed")
        multi_activations_outfeed_queue = MaybeOutfeedQueue(
                "multi_activations_outfeed",
                filters=process_filters(args.activations_filters))

        # Create a callback for the gradients
        gradients_outfeed_callback = OutfeedCallback(optimizer_outfeed_queue,
                                                     name="Gradients callback")

        # Create callbacks for the activations in the custom layers
        layer_outfeed_callback = OutfeedCallback(activations_outfeed_queue,
                                                 name="Single layer activations callback")

        multi_layer_outfeed_callback = OutfeedCallback(multi_activations_outfeed_queue,
                                                       name="Multi-layer activations callback")

        callbacks = [gradients_outfeed_callback]

        # Create an instance of the model.
        if args.model_type == "PipelineModel":
            model = create_pipeline_model(multi_activations_outfeed_queue,
                                          GRADIENT_ACCUMULATION_COUNT)
            callbacks += [multi_layer_outfeed_callback]
        elif args.model_type == "PipelineSequential":
            model = create_pipeline_sequential_model(multi_activations_outfeed_queue,
                                                     GRADIENT_ACCUMULATION_COUNT)
            callbacks += [multi_layer_outfeed_callback]
        elif args.model_type == "Sequential":
            if args.gradient_accumulation:
                gradient_accumulation_count = GRADIENT_ACCUMULATION_COUNT
            else:
                gradient_accumulation_count = 1
            model = create_sequential_model(activations_outfeed_queue,
                                            gradient_accumulation_count)
            callbacks += [layer_outfeed_callback]
        elif args.model_type == "Model":
            if args.gradient_accumulation:
                gradient_accumulation_count = GRADIENT_ACCUMULATION_COUNT
            else:
                gradient_accumulation_count = 1
            model = create_model(activations_outfeed_queue,
                                 gradient_accumulation_count)
            callbacks += [layer_outfeed_callback]
        else:
            print("Invalid model type: " + args.model_type)
            exit()

        # Get the training dataset.
        dataset = create_dataset()

        # Build the graph, passing an OutfeedOptimizer to enqueue selected gradients
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=OutfeedOptimizer(tf.keras.optimizers.SGD(),
                                                 optimizer_outfeed_queue,
                                                 outfeed_optimizer_mode=outfeed_optimizer_mode,
                                                 model=model))

        # Train the model, passing the callbacks to see the gradients and activations stats
        model.fit(dataset, callbacks=callbacks,
                  steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)

if __name__ == '__main__':
    main()
