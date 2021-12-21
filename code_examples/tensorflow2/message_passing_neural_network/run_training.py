# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf

from model import create_model
from absl import flags, app
import data
import uuid
import os
import numpy as np
import utils

from ogb.graphproppred import GraphPropPredDataset, Evaluator
import xpu

from utils import ThroughputCallback

flags.DEFINE_integer('batch_size', 128, 'compute batch size')
flags.DEFINE_integer('total_batch_size', None, 'total batch size (gradients will be accumulated)')

flags.DEFINE_integer('n_nodes', 24, 'nodes per graph')
flags.DEFINE_integer('n_edges', 50, 'edges per graph')

flags.DEFINE_integer('epochs', 100, 'maximum number of epochs to run for')
flags.DEFINE_float('lr', 2e-5, "learning rate")
flags.DEFINE_integer('reduce_lr_on_plateau_patience', 5, "if > 0, will decay LR on plateau of >= n epochs")
flags.DEFINE_float('reduce_lr_on_plateau_factor', 0.1, "factor by which to decay on plateau")
flags.DEFINE_integer('early_stopping_patience', 15, 'patience for early stopping')

flags.DEFINE_string('model_dir', './models', help='root directory for saving models')

FLAGS = flags.FLAGS


def main(_):
    tf.keras.mixed_precision.set_global_policy("float16" if FLAGS.dtype == 'float16' else "float32")

    dset_name = 'ogbg-molhiv'
    dataset = GraphPropPredDataset(name=dset_name, )
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    ds = data.get_tf_dataset(FLAGS.batch_size, [dataset[idx] for idx in train_idx], shuffle=True)
    val_ds = data.get_tf_dataset(FLAGS.batch_size, [dataset[idx] for idx in valid_idx], shuffle=False)
    strategy = xpu.configure_and_get_strategy()

    if FLAGS.total_batch_size is not None:
        gradient_accumulation_factor = FLAGS.total_batch_size // FLAGS.batch_size
    else:
        gradient_accumulation_factor = 1

    # pre-calculated number of steps per epoch (note: will vary somewhat for training, due to packing,
    #  but is found to be fairly consistent)
    steps = {
        32: (1195, 162, 148),
        64: (585, 80, 73),
        128: (288, 40, 37),
        256: (143, 20, 18)
    }
    try:
        steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = steps[FLAGS.batch_size]
    except KeyError:
        print("Batch size should have the number of steps defined")
        raise KeyError()

    # need the steps per epoch to be divisible by the gradient accumulation factor
    steps_per_epoch = gradient_accumulation_factor * (steps_per_epoch // gradient_accumulation_factor)

    # we apply a linear scaling rule for learning rate with batch size, which we benchmark against BS=128
    batch_size = FLAGS.total_batch_size or FLAGS.batch_size
    lr = FLAGS.lr * batch_size / 128

    with strategy.scope():
        model = create_model()
        utils.print_trainable_variables(model)

        losses = tf.keras.losses.BinaryCrossentropy()
        if FLAGS.opt.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr)
        elif FLAGS.opt.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise NotImplementedError()

        callbacks = []

        if not os.path.isdir(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        # randomly named directory
        model_dir = os.path.join(FLAGS.model_dir, str(uuid.uuid4()))

        print(f"Saving weights to {model_dir}")
        model_path = os.path.join(model_dir, 'model')

        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", verbose=1, save_best_only=True,
            save_weights_only=True, mode="min", save_freq="epoch")
        )

        callbacks.append(ThroughputCallback(
            samples_per_epoch=steps_per_epoch * FLAGS.batch_size * gradient_accumulation_factor))
        if FLAGS.reduce_lr_on_plateau_patience > 0:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', mode='min', factor=FLAGS.reduce_lr_on_plateau_factor,
                patience=FLAGS.reduce_lr_on_plateau_patience, min_lr=1e-8, verbose=1)
            )

        if FLAGS.early_stopping_patience > 0:
            print(f"Training will stop early after {FLAGS.early_stopping_patience} epochs without improvement.")
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=FLAGS.early_stopping_patience,
                    verbose=1, mode='min', baseline=None, restore_best_weights=False)
            )

        # weighted metrics are used because of the batch packing
        model.compile(optimizer=opt, loss=losses,
                      weighted_metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                      steps_per_execution=steps_per_epoch)

        # if the total batch size exceeds the compute batch size
        model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation_factor)

        model.fit(ds,
                  steps_per_epoch=steps_per_epoch,
                  epochs=FLAGS.epochs,
                  validation_data=val_ds,
                  validation_steps=val_steps_per_epoch,
                  callbacks=callbacks
                  )

        # we will use the official AUC evaluator from the OGB repo, not the keras one
        model.load_weights(model_path)
        print("Loaded best validation weights for evaluation")

        evaluator = Evaluator(name='ogbg-molhiv')
        for test_or_val, idx, steps in zip(
                ('validation', 'test'),
                (valid_idx, test_idx),
                (val_steps_per_epoch, test_steps_per_epoch)):
            prediction, ground_truth = get_predictions(model, dataset, idx, steps)
            result = evaluator.eval({'y_true': ground_truth[:, None], 'y_pred': prediction[:, None]})

            print(f'Final {test_or_val} ROC-AUC {result["rocauc"]:.3f}')


def get_predictions(model, dataset, dataset_idx, steps):
    """
    this function gets the model's predictions for a subset of the data (e.g. test, validation)

    :param model: the keras model with the weights loaded
    :param dataset: the ogb-molhiv data
    :param dataset_idx: idx for the chosen fold (e.g. the idx corresponding to test data)
    :param steps: number of steps to take to see all of the fold's data
    :return: ground_truth, prediction
    """
    data_subset = data.get_tf_dataset(FLAGS.batch_size, [dataset[idx] for idx in dataset_idx], shuffle=False)
    prediction = np.ravel(model.predict(data_subset, steps=steps))

    # we get the sample weights for the predictions -- we are only interested in the non-dummy data
    np_data_gen = data.np_batch_generator(FLAGS.n_nodes, FLAGS.n_edges, FLAGS.batch_size,
                                          [dataset[idx] for idx in dataset_idx], epochs=1, shuffle=False)
    sample_weights = []
    for batch in np_data_gen:
        # one of the labels is for the first dummy graph
        n_samples = np.sum(batch['labels'] != - 1)
        sample_weights.append([0] + [1] * n_samples + [0] * (FLAGS.batch_size - n_samples - 1))

    # select non-dummy graphs
    prediction = prediction[np.hstack(sample_weights) == 1]
    ground_truth = np.ravel([dataset[idx][1] for idx in dataset_idx])
    return prediction, ground_truth


if __name__ == '__main__':
    app.run(main)
