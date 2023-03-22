# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import tensorflow as tf


def pretraining_accuracy_fn(labels, logits):
    """
    Compute Masked Language Model (MLM) and Next Sentence Prediction (NSP) accuracy.
    MLM:
        :param labels: Categorical tensor of shape: (batch size, masked token length)
        :param logits: Tensor of shape (batch size, masked token length, vocab size).
    NSP:
        :param labels: Binary tensor of shape (batch size, 1).
        :param logits: Tensor of shape (batch size, 2).
    :return: Scalar value corresponding to the MLM and NSP accuracy after averaging the batch.
    """
    #  check if logits are from MLM head
    if len(tf.shape(logits)) == 3:
        mask = tf.not_equal(labels, 0)
        mask = tf.cast(mask, tf.float16)
        labels = tf.cast(labels, tf.int32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        results = tf.cast(tf.argmax(log_probs, -1), dtype=tf.int32)
        predictions = tf.cast(tf.equal(results, labels), dtype=tf.float16)
        predictions = tf.cast(predictions * mask, dtype=tf.float32)
        acc_total = tf.reduce_sum(predictions)
        total_samples = tf.cast(tf.reduce_sum(mask), dtype=tf.float32)
        acc = acc_total / total_samples

    # check if logits are from NSP head
    elif len(tf.shape(logits)) == 2:
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.cast(labels, tf.int32)
        next_sentence_labels = tf.reshape(labels, [-1])
        predict_labels = tf.cast(tf.argmax(log_probs, -1), dtype=tf.int32)
        predict_labels = tf.reshape(predict_labels, [-1])
        acc = tf.cast(tf.equal(predict_labels, next_sentence_labels), dtype=tf.float32)
        acc = tf.reduce_mean(acc)
    return acc


def classification_accuracy_fn(labels, logits):
    """
    Compute the classification accuracy.
    :return: Scalar value corresponding to the accuracy after averaging the batch.
    """
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    predict_labels = tf.reshape(tf.cast(tf.argmax(log_probs, -1), dtype=tf.int32), [-1])
    acc = tf.cast(tf.equal(predict_labels, labels), dtype=tf.float32)
    acc = tf.reduce_mean(acc)

    return acc
