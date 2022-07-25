# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging

import tensorflow as tf
import tensorflow_addons as tfa

from utilities.constants import AdjacencyForm, MASKED_LABEL_VALUE, Task
from utilities.metric_enqueuer import wrap_loss_in_enqueuer


def get_loss_and_metrics(task,
                         num_labels,
                         adjacency_form,
                         metrics_precision,
                         enable_loss_outfeed=True):
    """Given the dataset type and the number of labels, returns
       the loss function, accuracy and F1 score metrics."""
    if task == Task.BINARY_MULTI_LABEL_CLASSIFICATION:
        logging.info("Using loss and accuracy for binary multi"
                     " label classification task.")
        return get_loss_accuracy_f1score_for_binary_case(
            num_labels,
            adjacency_form,
            metrics_precision,
            enable_loss_outfeed=enable_loss_outfeed)
    elif task == Task.MULTI_CLASS_CLASSIFICATION:
        logging.info("Using loss and accuracy for multi"
                     " class classification task.")
        return get_loss_accuracy_f1score_for_categorical_case(
            num_labels,
            adjacency_form,
            metrics_precision,
            enable_loss_outfeed=enable_loss_outfeed)
    raise ValueError(f"Provided task must be one of type {Task}")


def get_loss_accuracy_f1score_for_binary_case(num_labels,
                                              adjacency_form,
                                              metrics_precision,
                                              enable_loss_outfeed=True):
    """Given the number of labels, returns the loss function, accuracy and
    F1 score metrics for binary multi label classification."""
    loss_fn_class = get_masked_loss_class(tf.keras.losses.BinaryCrossentropy,
                                          activation=tf.math.sigmoid)
    if enable_loss_outfeed:
        # In order to obtain the actual instantaneous loss, we wrap
        # the loss class with functionality to enqueue the loss to
        # an outfeed queue.
        loss_fn_class = wrap_loss_in_enqueuer(loss_fn_class,
                                              ["loss_instantaneous"])

    loss = loss_fn_class(adjacency_form=adjacency_form,
                         from_logits=False,
                         reduction=tf.keras.losses.Reduction.NONE,
                         name="loss_epoch_avg",
                         dtype=metrics_precision)

    accuracy_metric_class = get_masked_accuracy_class(
        tf.keras.metrics.BinaryAccuracy,
        activation=tf.math.sigmoid)
    accuracy = accuracy_metric_class(adjacency_form=adjacency_form,
                                     name="accuracy_epoch_avg",
                                     dtype=metrics_precision)

    f1_score_macro = MaskedF1Score(
        num_classes=num_labels,
        average="macro",
        metrics_precision=metrics_precision,
        adjacency_form=adjacency_form,
        threshold=0.5,
        activation=tf.math.sigmoid,
        name="f1_score_macro_epoch_avg")
    f1_score_micro = MaskedF1Score(
        num_classes=num_labels,
        average="micro",
        metrics_precision=metrics_precision,
        adjacency_form=adjacency_form,
        threshold=0.5,
        activation=tf.math.sigmoid,
        name="f1_score_micro_epoch_avg")

    return loss, accuracy, f1_score_macro, f1_score_micro


def get_loss_accuracy_f1score_for_categorical_case(num_labels,
                                                   adjacency_form,
                                                   metrics_precision,
                                                   enable_loss_outfeed=True):
    """Given the number of labels, returns the loss function, accuracy and F1
    score metrics for multi class classification."""

    loss_fn_class = get_masked_loss_class(
        tf.keras.losses.SparseCategoricalCrossentropy)

    if enable_loss_outfeed:
        # In order to obtain the actual instantaneous loss, we wrap
        # the loss class with functionality to enqueue the loss to
        # an outfeed queue.
        loss_fn_class = wrap_loss_in_enqueuer(loss_fn_class,
                                              ["loss_instantaneous"])

    loss = loss_fn_class(adjacency_form=adjacency_form,
                         from_logits=True,
                         reduction=tf.keras.losses.Reduction.NONE,
                         name="loss_epoch_avg",
                         dtype=metrics_precision)

    accuracy_metric_class = get_masked_accuracy_class(
        tf.keras.metrics.SparseCategoricalAccuracy)
    accuracy = accuracy_metric_class(adjacency_form=adjacency_form,
                                     name="accuracy_epoch_avg",
                                     dtype=metrics_precision)

    f1_score_macro = MaskedF1Score(
        num_classes=num_labels,
        average="macro",
        metrics_precision=metrics_precision,
        adjacency_form=adjacency_form,
        name="f1_score_macro_epoch_avg",
        labels_to_one_hot=True)
    f1_score_micro = MaskedF1Score(
        num_classes=num_labels,
        average="micro",
        metrics_precision=metrics_precision,
        adjacency_form=adjacency_form,
        name="f1_score_micro_epoch_avg",
        labels_to_one_hot=True)

    return loss, accuracy, f1_score_macro, f1_score_micro


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


def get_masked_loss_class(loss_class,
                          activation=tf.keras.activations.linear):
    """Wraps and returns a loss function class with functionality
    to regenerate and apply the mask to the result. An activation can
    also be supplied, such as sigmoid or softmax"""
    class MaskedLoss(loss_class):

        def __init__(self, adjacency_form, dtype=tf.float32, *args, **kwargs):
            if kwargs["reduction"] != tf.keras.losses.Reduction.NONE:
                raise ValueError("In order to correctly apply the mask"
                                 " the reduction type must be NONE.")
            super().__init__(*args, **kwargs)
            self.adjacency_form = adjacency_form
            self.dtype = dtype

        def call(self, y_true, y_pred):
            """
            Returns the loss.
            :param y_true: The labels, in our case this is shape
                (num_nodes, label_size)
                Note that for the case of Sparse Tuple, the shape will be
                (micro_batch_size, num_nodes, label_size), in which micro_batch_size = 1
            :param y_pred: The predictions, the same shape as labels.
            """
            # Squeeze labels on batch dimension
            if self.adjacency_form == AdjacencyForm.SPARSE_TUPLE:
                y_true = tf.squeeze(y_true, axis=0)
            # Cast to desired precision
            y_true = tf.cast(y_true, dtype=self.dtype)
            y_pred = tf.cast(y_pred, dtype=self.dtype)
            # Regenerate the mask from the labels, shape (num_nodes)
            mask = get_mask_from_labels(y_true)
            # Set the labels at masked positions to `0` to avoid issues
            # using `-1` in the CategoricalCrossentropy loss.
            masked_labels = set_masked_elements_to_zero(y_true, mask)
            # Apply activation to the prediction
            y_pred = activation(y_pred)
            # Call the CategoricalCrossentropy loss
            loss = super().call(masked_labels, y_pred)
            # Apply the mask to the loss
            mask_cast = tf.cast(mask, loss.dtype)
            loss *= mask_cast
            return tf.math.divide_no_nan(tf.reduce_sum(loss),
                                         tf.reduce_sum(mask_cast))

    return MaskedLoss


def get_masked_accuracy_class(accuracy_metric_class,
                              activation=tf.keras.activations.linear):
    """Wraps and returns an accuracy metric class with functionality
    to regenerate and apply the mask to the result. An activation can
    also be supplied, such as sigmoid or softmax"""

    class MaskedAccuracy(accuracy_metric_class):

        def __init__(self, adjacency_form, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.adjacency_form = adjacency_form

        def update_state(self, y_true, y_pred, sample_weight=None):
            # Squeeze labels on batch dimension
            if self.adjacency_form == AdjacencyForm.SPARSE_TUPLE:
                y_true = tf.squeeze(y_true, axis=0)
            # Regenerate the mask from the labels, shape (num_nodes)
            mask = get_mask_from_labels(y_true)
            # Apply activation to the prediction
            y_pred = activation(y_pred)
            # Update the metric state with the mask as sample_weight
            super().update_state(y_true,
                                 y_pred,
                                 sample_weight=tf.cast(mask, y_pred.dtype))
    return MaskedAccuracy


class MaskedF1Score(tfa.metrics.F1Score):
    """Child class of TensorFlow F1 score suitable with masking in this
    application. It regenerates the mask from the labels and
    applies it to the F1 score."""

    def __init__(self,
                 metrics_precision,
                 adjacency_form,
                 activation=tf.keras.activations.linear,
                 labels_to_one_hot=False,
                 *args,
                 **kwargs):
        super().__init__(dtype=metrics_precision,
                         *args,
                         **kwargs)
        self.metrics_precision = metrics_precision
        self.activation = activation
        self.labels_to_one_hot = labels_to_one_hot
        self.adjacency_form = adjacency_form

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Squeeze labels on batch dimension
        if self.adjacency_form == AdjacencyForm.SPARSE_TUPLE:
            y_true = tf.squeeze(y_true, axis=0)
        # Regenerate the mask from the labels, shape (num_nodes)
        mask = get_mask_from_labels(y_true)
        if self.labels_to_one_hot:
            y_true = tf.one_hot(tf.reshape(y_true, [-1]),
                                depth=self.num_classes,
                                dtype=self.metrics_precision)
        # Apply activation to the prediction
        y_pred = self.activation(y_pred)
        # Update the metric state with the mask as sample_weight
        super().update_state(y_true,
                             y_pred,
                             sample_weight=tf.cast(mask, self.metrics_precision))
