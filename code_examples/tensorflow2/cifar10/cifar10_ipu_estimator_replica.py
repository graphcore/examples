# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

from data import CIFAR10_Data
from model import get_model
from utils import parse_params, create_ipu_config


def estimator_model(features, labels, mode):
    model = get_model(logits_only=True)
    logits = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    preds = tf.argmax(input=logits, axis=-1)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    # Wrap the optimiser for sharding
    optimiser = ipu.optimizers.CrossReplicaOptimizer(optimizer)
    train_op = optimiser.minimize(loss)
    eval_accuracy, eval_op = tf.compat.v1.metrics.accuracy(labels, preds)
    metric_ops = {"accuracy": (eval_accuracy, eval_op)}

    return tf.estimator.EstimatorSpec(mode=mode, predictions=preds, loss=loss, train_op=train_op, eval_metric_ops=metric_ops)


if __name__ == '__main__':
    opts = parse_params(enable_multi_ipu=True)
    print("Loading the data...")
    data = CIFAR10_Data()

    print("Initialize the model")
    test_steps = len(data.y_test) // opts.batch_size
    training_steps = 5 * test_steps
    config = create_ipu_config(training_steps, test_steps, num_replicas=opts.ipus)
    ipu_estimator = ipu.ipu_estimator.IPUEstimator(config=config, model_fn=estimator_model)

    print("Training...")
    ipu_estimator.train(partial(data.get_train_datagenerator, opts.batch_size), steps=training_steps*opts.epochs)

    print("Check the result...")
    result = ipu_estimator.evaluate(partial(data.get_test_datagenerator, opts.batch_size), steps=test_steps)
    print("Validation accuracy: {}%".format(100.0 * result['accuracy']))
    print("Validation loss: {}".format(result['loss']))
