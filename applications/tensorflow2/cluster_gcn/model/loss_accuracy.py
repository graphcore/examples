# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python import ipu

from utilities.constants import MASKED_LABEL_VALUE, Task
from utilities.metric_enqueuer import wrap_metric_in_enqueuer


def get_loss_accuracy_f1score(task, num_labels, metrics_precision):
    """Given the dataset type and the number of labels, returns
       the loss function, accuracy and F1 score metrics."""
    if task == Task.BINARY_MULTI_LABEL_CLASSIFICATION:
        logging.info("Using loss and accuracy for binary multi"
                     " label classification task.")
        return get_loss_accuracy_f1score_for_binary_case(num_labels, metrics_precision)
    elif task == Task.MULTI_CLASS_CLASSIFICATION:
        logging.info("Using loss and accuracy for multi"
                     " class classification task.")
        return get_loss_accuracy_f1score_for_categorical_case(num_labels, metrics_precision)
    raise ValueError(f"Provided task must be one of type {Task}")


def get_loss_accuracy_f1score_for_binary_case(num_labels, metrics_precision):
    """Given the number of labels, returns the loss function,
       accuracy and F1 score metrics for binary multi label
       classification."""
    # In order to obtain the actual instantaneous loss, we create
    # an enqueue it to an outfeed queue.
    loss_fn = wrap_metric_in_enqueuer(MaskedBinaryMultiLabelCrossentropy,
                                      ipu.ipu_outfeed_queue.IPUOutfeedQueue(),
                                      ["loss_instantaneous"])
    loss = loss_fn(from_logits=False,
                   reduction=tf.keras.losses.Reduction.NONE,
                   name="loss_epoch_avg")

    accuracy = MaskedBinaryMultiLabelAccuracy(name="accuracy_epoch_avg")

    f1_score = MaskedBinaryMultiLabelF1Score(num_classes=num_labels,
                                             average="macro",
                                             metrics_precision=metrics_precision,
                                             name="f1_score_epoch_avg")
    return loss, accuracy, f1_score


def get_loss_accuracy_f1score_for_categorical_case(num_labels, metrics_precision):
    """Given the number of labels, returns the loss function,
       accuracy and F1 score metrics for multi class classification."""
    # In order to obtain the actual instantaneous loss, we create
    # an enqueue it to an outfeed queue.
    loss_fn = wrap_metric_in_enqueuer(MaskedCategoricalCrossentropy,
                                      ipu.ipu_outfeed_queue.IPUOutfeedQueue(),
                                      ["loss_instantaneous"])
    loss = loss_fn(from_logits=True,
                   reduction=tf.keras.losses.Reduction.NONE,
                   name="loss_epoch_avg")

    accuracy = MaskedCategoricalAccuracy(name="accuracy_epoch_avg")

    f1_score = MaskedCategoricalF1Score(num_classes=num_labels,
                                        average="macro",
                                        metrics_precision=metrics_precision,
                                        name="f1_score_epoch_avg")
    return loss, accuracy, f1_score


def get_mask_from_labels(labels):
    """Given the labels of shape (num_nodes, label_size), returns an
       array representing the mask for each node, size (num_nodes,)."""
    # If there is a masked label value in the first element of each
    # node we know that node is masked.
    return tf.not_equal(labels[:, 0], MASKED_LABEL_VALUE)


def set_masked_elements_to_zero(labels, mask):
    """Given the labels of shape (num_nodes, label_size) and mask
       of shape (num_nodes,), returns the labels with its masked nodes
       set to zero."""
    # Repeat the mask in the labels dimension so it is the same shape
    # as labels
    mask_repeated = tf.repeat(tf.reshape(mask, (-1, 1)),
                              labels.shape[-1],
                              axis=1)
    # Apply mask to labels to make the masked values `0`.
    labels = labels * tf.cast(mask_repeated, labels.dtype)
    return labels


class MaskedBinaryMultiLabelCrossentropy(tf.keras.losses.BinaryCrossentropy):
    """Child class of TensorFlow Keras BinaryCrossentropy that
       regenerates the mask from the labels and applies loss function
       it to the loss function. Only supports `None` reduction type."""

    def __init__(self, *args, **kwargs):
        if kwargs["reduction"] != tf.keras.losses.Reduction.NONE:
            raise ValueError("In order to correctly apply the mask"
                             " the reduction type must be NONE.")
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        """
        Returns the loss.
        :param y_true: The labels, in our case this is shape
            (num_nodes, label_size)
        :param y_pred: The predictions, the same shape as labels.
        """
        # Regenerate the mask from the labels, shape (num_nodes)
        mask = get_mask_from_labels(y_true)
        # Set the labels at masked positions to `0` to avoid issues
        # using `-1` in the CategoricalCrossentropy loss.
        masked_labels = set_masked_elements_to_zero(y_true, mask)
        # Apply sigmoid to the prediction as we are doing multi label
        # prediction
        y_pred = tf.math.sigmoid(y_pred)
        # Apply the mask to the loss
        loss = super().call(masked_labels, y_pred)
        mask_cast = tf.cast(mask, loss.dtype)
        loss *= mask_cast
        return tf.math.divide_no_nan(tf.reduce_sum(loss),
                                     tf.reduce_sum(mask_cast))


class MaskedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    """Child class of TensorFlow Keras CategoricalCrossentropy that
       regenerates the mask from the labels and applies loss function
       it to the loss function. Only supports `None` reduction type."""

    def __init__(self, *args, **kwargs):
        if kwargs["reduction"] != tf.keras.losses.Reduction.NONE:
            raise ValueError("In order to correctly apply the mask"
                             " the reduction type must be NONE.")
        super().__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        """
        Returns the loss.
        :param y_true: The labels, in our case this is shape
            (num_nodes, label_size)
        :param y_pred: The predictions, the same shape as labels.
        """
        # Regenerate the mask from the labels, shape (num_nodes)
        mask = get_mask_from_labels(y_true)
        # Set the labels at masked positions to `0` to avoid issues
        # using `-1` in the CategoricalCrossentropy loss.
        masked_labels = set_masked_elements_to_zero(y_true, mask)
        # Call the CategoricalCrossentropy loss
        loss = super().call(masked_labels, y_pred)
        # Apply the mask to the loss
        mask_cast = tf.cast(mask, loss.dtype)
        loss *= mask_cast
        return tf.math.divide_no_nan(tf.reduce_sum(loss),
                                     tf.reduce_sum(mask_cast))


class MaskedBinaryMultiLabelAccuracy(tf.keras.metrics.BinaryAccuracy):
    """Child class of TensorFlow BinaryAccuracy that regenerates the
       mask from the labels and applies it to the accuracy."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_mask_from_labels(y_true)
        # Apply sigmoid to the prediction as we are doing multi label
        # prediction
        y_pred = tf.math.sigmoid(y_pred)
        super().update_state(y_true,
                             y_pred,
                             sample_weight=tf.cast(mask, y_pred.dtype))


class MaskedCategoricalAccuracy(tf.keras.metrics.CategoricalAccuracy):
    """Child class of TensorFlow CategoricalAccuracy that regenerates the
       mask from the labels and applies it to the accuracy."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_mask_from_labels(y_true)
        # No need to apply softmax as the argmax of logits and probabilities
        # will be the same.
        super().update_state(y_true,
                             y_pred,
                             sample_weight=tf.cast(mask, y_pred.dtype))


class MaskedBinaryMultiLabelF1Score(tfa.metrics.F1Score):
    """Child class of TensorFlow F1 score for binary multi label
       classification. It regenerates the mask from the labels and
       applies it to the F1 score."""

    def __init__(self,
                 num_classes,
                 metrics_precision,
                 average,
                 name="f1_score"):
        super().__init__(num_classes=num_classes,
                         average=average,
                         threshold=0.5,
                         name=name,
                         dtype=metrics_precision)
        self.metrics_precision = metrics_precision

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_mask_from_labels(y_true)
        # Apply sigmoid to the prediction as we are doing multi label
        # prediction
        y_pred = tf.math.sigmoid(y_pred)
        super().update_state(y_true,
                             y_pred,
                             sample_weight=tf.cast(mask, self.metrics_precision))


class MaskedCategoricalF1Score(tfa.metrics.F1Score):
    """Child class of TensorFlow F1 score for classification. It
       regenerates the mask from the labels and applies it to the
       F1 score."""

    def __init__(self,
                 num_classes,
                 metrics_precision,
                 average,
                 name="f1_score"):
        super().__init__(num_classes=num_classes,
                         average=average,
                         threshold=None,
                         name=name,
                         dtype=metrics_precision)
        self.metrics_precision = metrics_precision

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_mask_from_labels(y_true)
        # No need to apply softmax if the threshold is set to None. In
        # this case the argmax would be converted to 1, and the rest 0.
        super().update_state(y_true,
                             y_pred,
                             sample_weight=tf.cast(mask, self.metrics_precision))
