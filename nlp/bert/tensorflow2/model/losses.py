# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import tensorflow as tf
from keras.losses import LossFunctionWrapper


def mlm_loss(labels, logits):
    """
    Compute Masked Language Model (MLM) loss.
    A custom loss function is used here rather than sparse categorical
    cross entropy, as using sparse categorical cross entropy converges to
    a higher loss. The following is also more robust to corrupt data, if
    label values outside the allowable range.
    :param labels: Tensor of shape: (batch size, max_predictions_per_seq)
    :param logits: Tensor of shape (batch size, max_predictions_per_seq, vocab size).
    :return: Scalar value corresponding to the MLM loss after averaging the batch.
    """
    micro_batch_size, max_predictions_per_seq, vocab_size = logits.shape
    logits = tf.reshape(logits, (micro_batch_size * max_predictions_per_seq, vocab_size))

    log_probs = tf.nn.log_softmax(logits)

    label_ids = tf.reshape(labels, (-1,))
    one_hot_labels = tf.one_hot(label_ids, depth=vocab_size, dtype=log_probs.dtype)
    per_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=-1)

    mask = tf.not_equal(labels, 0)
    mask = tf.cast(tf.reshape(mask, (-1,)), dtype=per_loss.dtype)

    numerator = tf.cast(tf.reduce_sum(mask * per_loss), dtype=tf.float32)
    denominator = tf.cast(tf.reduce_sum(mask), dtype=tf.float32)
    return numerator / denominator


def nsp_loss(labels, logits):
    """
    Compute Next Sentence Prediction (NSP) loss.
    :param labels: Binary tensor of shape (batch size, 1).
    :param logits: Tensor of shape (batch size, 2).
    :return: Scalar value corresponding to the NSP loss after averaging the batch.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    loss = loss_fn(labels, logits)
    return tf.cast(tf.reduce_mean(loss), dtype=tf.float32)


def question_answering_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    loss = loss_fn(labels, logits)
    return tf.reduce_mean(loss)


def classification_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    loss = loss_fn(labels, logits)
    return tf.reduce_mean(loss)


def classification_loss_regression(labels, logits):
    loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    loss = loss_fn(labels, logits)
    return tf.reduce_mean(loss)


class NSPLossFunction(LossFunctionWrapper):
    """Keras Loss wrapper for the NSP Loss loss between the labels and predictions."""

    def __init__(self, name="NSP_LossCalculation"):
        super().__init__(nsp_loss, name=name)


class MLMLossFunction(LossFunctionWrapper):
    """Keras Loss wrapper for the MLM Loss loss between the labels and predictions."""

    def __init__(self, name="MLM_LossCalculation"):
        super().__init__(mlm_loss, name=name)


class QuestionAnsweringLossFunction(LossFunctionWrapper):
    """Keras loss wrapper for the question-answer loss."""

    def __init__(self, name="QuestionAnswerLossCalculation"):
        super().__init__(question_answering_loss, name=name)


class ClassificationLossFunction(LossFunctionWrapper):
    """Keras loss wrapper for the classification loss."""

    def __init__(self, name="ClassificationLossFunction"):
        super().__init__(classification_loss, name=name)


class ClassificationLossFunctionRegression(LossFunctionWrapper):
    """Keras loss wrapper for the classification loss for regression tasks."""

    def __init__(self, name="ClassificationLossFunctionRegression"):
        super().__init__(classification_loss_regression, name=name)
