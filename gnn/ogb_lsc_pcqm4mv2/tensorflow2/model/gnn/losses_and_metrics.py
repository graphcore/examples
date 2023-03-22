# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from keras import backend


def MaskedMeanAbsoluteError(y_true, y_pred, transform=lambda x: x, padding_value=-1):
    # masked mean absolute error where padding elements are marked by -1
    y_true = tf.cast(y_true, tf.float32)  # defensive cast
    y_pred = tf.cast(y_pred, tf.float32)  # defensive cast
    mask = tf.where(y_true == padding_value, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))
    y_true = transform(y_true)
    y_pred = transform(y_pred)
    return backend.sum(tf.abs(y_pred - y_true) * mask) / backend.sum(mask)


def MaskedMeanSquaredError(y_true, y_pred, transform=lambda x: x, padding_value=-1, zero_value=None):
    # masked mean squared error where padding elements are marked by -1
    y_true = tf.cast(y_true, tf.float32)  # defensive cast
    y_pred = tf.cast(y_pred, tf.float32)  # defensive cast
    padding_mask = tf.where(y_true == padding_value, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))
    if zero_value is not None:
        zero_mask = tf.where(y_true == zero_value, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))
    y_true = transform(y_true)
    y_pred = transform(y_pred)
    out = tf.square(y_pred - y_true) * padding_mask
    if zero_value is not None:
        out = out * zero_mask
    return backend.sum(out) / backend.sum(padding_mask)


def CosineSimilarityError(y_true, y_pred, padding_value=-1, zero_value=None):
    y_true = tf.cast(y_true, tf.float32)  # [batch, nodes, 3]
    y_pred = tf.cast(y_pred, tf.float32)  # [batch, nodes, 3]
    padding_mask = tf.where(y_true == padding_value, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))[
        :, :, 0
    ]  # [batch, nodes]
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
    loss = (1.0 - cosine_loss(y_true, y_pred)) * padding_mask  # [batch, nodes]
    if zero_value is not None:
        zero_mask = tf.where(y_true == zero_value, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))[:, :, 0]
        loss *= zero_mask
    # Scale over all the nodes and graphs (excluding padding ones)
    return backend.sum(loss) / tf.maximum(backend.sum(padding_mask), 1.0)


def LossNoisyNodes(labels, pred, loss_weight=1.0, mode="nodes", method="combined_softmax", vocab_size=[]):
    pred = tf.cast(pred, tf.float32)  # defensive cast
    # labels has size [batch, nodes/edges, num feature categories]
    # preds has size [batch, nodes/edges, sum possible features]
    assert len(vocab_size) == labels.get_shape().as_list()[-1]
    assert sum(vocab_size) == pred.get_shape().as_list()[-1]

    mask = tf.where(labels == -1, tf.cast(0, pred.dtype), tf.cast(1, pred.dtype))
    mask = tf.unstack(mask, axis=-1)[0]  # mask has size [batch, nodes/edges] in float32

    labels = tf.cast(labels, tf.int32)
    labels = tf.unstack(labels, axis=-1)
    # convert labels to one_hot representation
    labels = [tf.one_hot(l, v, axis=-1) for l, v in zip(labels, vocab_size)]

    if method == "combined_softmax":
        # do a single softmax over all options (same as DeepMind)

        # conat to match pred shape
        labels = tf.concat(labels, axis=-1)
        labels = tf.cast(labels, tf.float32)

        # mask labels
        labels = labels * tf.broadcast_to(mask[..., tf.newaxis], labels.shape)

        # normalise last dim to sum to 1
        labels = labels / tf.maximum(tf.reduce_sum(labels, axis=-1), 1)[..., tf.newaxis]
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred, axis=-1)
    elif method == "split_softmax":
        # split pred to match labels
        pred = tf.split(pred, vocab_size, axis=-1)

        losses = tf.zeros_like(mask)
        for label, p in zip(labels, pred):
            label = tf.cast(label, tf.float32)
            # mask labels
            label = label * tf.broadcast_to(mask[..., tf.newaxis], label.shape)
            losses += [tf.nn.softmax_cross_entropy_with_logits(label, p, axis=-1)]
        losses /= len(labels)  # take mean over different categories
    else:
        raise ValueError()

    # masked mean
    out = loss_weight * backend.sum(losses * mask) / backend.sum(mask)
    return out


def batch_size(y_true, y_pred, cfg):
    # masked mean absolute error where padding elements are marked by -1
    y_true = tf.cast(y_true, tf.float32)  # defensive cast
    y_pred = tf.cast(y_pred, tf.float32)  # defensive cast
    mask = tf.where(y_true == -1, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))
    return cfg.ipu_opts.replicas * cfg.ipu_opts.gradient_accumulation_factor * backend.sum(mask)


def label_mean(y_true, y_pred):
    # masked mean absolute error where padding elements are marked by -1
    y_true = tf.cast(y_true, tf.float32)  # defensive cast
    mask = tf.where(y_true == -1, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))

    return backend.sum(y_true * mask) / backend.sum(mask)


def pred_mean(y_true, y_pred):
    # masked mean absolute error where padding elements are marked by -1
    y_true = tf.cast(y_true, tf.float32)  # defensive cast
    mask = tf.where(y_true == -1, tf.cast(0, y_true.dtype), tf.cast(1, y_true.dtype))
    y_pred = tf.cast(y_pred, tf.float32)  # defensive cast
    return backend.sum(y_pred * mask) / backend.sum(mask)


def prepare_metric_inputs(mask, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    if mask.dtype == tf.int32:
        mask = tf.cast(mask, tf.float32)
    else:
        mask = tf.where(mask == -1, tf.cast(0, y_pred.dtype), tf.cast(1, y_pred.dtype))

    y_pred = mask * y_pred
    return mask, y_pred


def max_abs(mask, y_pred):
    mask, y_pred = prepare_metric_inputs(mask, y_pred)
    return tf.reduce_max(tf.abs(y_pred))


def mean_abs(mask, y_pred):
    mask, y_pred = prepare_metric_inputs(mask, y_pred)
    scale = tf.cast(tf.size(y_pred), tf.float32) / tf.reduce_sum(tf.broadcast_to(mask, y_pred.shape))
    return scale * tf.reduce_mean(tf.abs(y_pred))


def mean(mask, y_pred):
    mask, y_pred = prepare_metric_inputs(mask, y_pred)
    scale = tf.cast(tf.size(y_pred), tf.float32) / tf.reduce_sum(tf.broadcast_to(mask, y_pred.shape))
    return scale * tf.reduce_mean(y_pred)


def var(mask, y_pred):
    mask, y_pred = prepare_metric_inputs(mask, y_pred)
    mean, var = tf.nn.moments(y_pred, axes=[0, 1, 2])
    # must correct for padding nodes
    scale = tf.cast(tf.size(y_pred), tf.float32) / tf.reduce_sum(tf.broadcast_to(mask, y_pred.shape))
    real_mean = scale * mean
    mean_sqr_true = scale * (var + (mean**2))
    var_true = mean_sqr_true - (real_mean**2)
    return var_true


def get_debug_metrics(names, metrics):
    metric_fns = []
    if "max_abs" in metrics:
        metric_fns += [max_abs]
    if "mean_abs" in metrics:
        metric_fns += [mean_abs]
    if "mean" in metrics:
        metric_fns += [mean]
    if "var" in metrics:
        metric_fns += [var]
    if "var_corr" in metrics:
        metric_fns += [var_corr]

    return {n: metric_fns for n in names}


def set_debug_labels(layer_names, node_mask=None, edge_mask=None, global_mask=None):
    labels = []
    node_mask = tf.cast(node_mask, tf.int32)
    edge_mask = tf.cast(edge_mask, tf.int32)
    global_mask = tf.cast(global_mask, tf.int32)
    for n in layer_names:
        if "Nodes" in n:
            labels += [node_mask]
        elif "Edges" in n:
            labels += [edge_mask]
        elif "Globals" in n:
            labels += [global_mask]
        else:
            raise ValueError(f"No label set for layer {n}")
    return labels
