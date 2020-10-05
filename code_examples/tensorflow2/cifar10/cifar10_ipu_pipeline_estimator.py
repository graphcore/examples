# Copyright 2020 Graphcore Ltd.
from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

from data import CIFAR10_Data
from model import get_staged_model
from utils import parse_params, create_ipu_config


def estimator_model(opts, mode):
    stage0, stage1 = get_staged_model(is_training=mode == tf.estimator.ModeKeys.TRAIN, model_shard_position=opts.model_shard_position)

    def final_stage(features, labels):
        logits, labels = stage1(features, labels)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
        preds = tf.argmax(input=logits, axis=-1, output_type=tf.int32)
        return loss, preds, labels

    def optimizer_fn(loss, predictions, labels):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        return ipu.ops.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

    def metric_fn(loss, predictions, labels):
        return {"loss": loss, "accuracy": tf.compat.v1.metrics.accuracy(labels, predictions)}
    stages = [stage0, final_stage]
    return ipu.ipu_pipeline_estimator.IPUPipelineEstimatorSpec(mode=mode,
                                                               computational_stages=stages,
                                                               pipeline_depth=opts.pipeline_depth,
                                                               optimizer_function=optimizer_fn,
                                                               eval_metrics_fn=metric_fn,
                                                               pipeline_schedule=None)


if __name__ == '__main__':
    opts = parse_params(enable_multi_ipu=True, enable_pipelining=True)
    print("Loading the data...")
    data = CIFAR10_Data()

    print("Initialize the model")
    test_steps = len(data.y_test) // (opts.batch_size * opts.pipeline_depth)
    training_steps = 5 * test_steps
    config = create_ipu_config(training_steps, test_steps, num_shards=opts.ipus)
    ipu_estimator = ipu.ipu_pipeline_estimator.IPUPipelineEstimator(config=config, model_fn=partial(estimator_model, opts))

    ipu_estimator.train(partial(data.get_train_datagenerator, opts.batch_size), steps=training_steps*opts.epochs)

    print("Check the result...")
    result = ipu_estimator.evaluate(partial(data.get_test_datagenerator, opts.batch_size), steps=test_steps)
    print("Validation accuracy: {}%".format(100.0 * result['accuracy']))
    print("Validation loss: {}".format(result['loss']))
