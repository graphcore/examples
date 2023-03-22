# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf

import xpu


class GenericAggregator(tf.keras.layers.Layer):
    def __init__(self, aggregators, name="GenericAggregator", gather_scatter_method="grouped"):
        super(GenericAggregator, self).__init__(name=name)
        self.aggregators = aggregators
        self.gather_scatter_method = gather_scatter_method

    def build(self, input_shape):
        if "softmax" in self.aggregators:
            self.beta = self.add_weight("beta", shape=(), initializer="ones", trainable=True)

    def call(self, data, indices, num_segments, stable_var=True, training=True, *args, **kwargs):
        outputs = []
        if "sum" in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_sum,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                )
            ]

        if "max" in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_max,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                )
            ]

        if "min" in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_min,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                )
            ]

        if "mean" in self.aggregators and "var" not in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_mean,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                )
            ]
        if "var" in self.aggregators and "mean" not in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_var,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                    stable_var=stable_var,
                )
            ]
        # if both 'mean' and 'var' are needed for aggregators, we could
        # avoid calculating scatter_mean twice
        if "mean" in self.aggregators and "var" in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_mean_and_var,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                    stable_var=stable_var,
                )
            ]

        if "std" in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_std,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                )
            ]

        if "sqrtN" in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_sqrtN,
                    data,
                    indices,
                    num_segments=num_segments,
                    gather_scatter_method=self.gather_scatter_method,
                )
            ]

        if "softmax" in self.aggregators:
            outputs += [
                xpu.call_outlined_function(
                    _scatter_softmax,
                    data,
                    indices,
                    num_segments=num_segments,
                    beta=self.beta,
                    gather_scatter_method=self.gather_scatter_method,
                )
            ]
        return tf.concat(outputs, axis=-1)


def _gather(data, indices, gather_scatter_method="grouped"):
    if gather_scatter_method == "grouped":
        x = xpu.grouped_gather(data, indices)
    elif gather_scatter_method == "dense":
        x = tf.matmul(tf.cast(indices, data.dtype), data)
    elif gather_scatter_method == "debug":
        x = tf.stack([tf.gather(data[i], indices[i]) for i in range(data.shape[0])])
    else:
        raise ValueError(f"gather_scatter method {gather_scatter_method} is invalid.")
    return x


def gather(data, indices, gather_scatter_method="grouped"):
    return xpu.call_outlined_function(_gather, data, indices, gather_scatter_method=gather_scatter_method)


def _scatter_sum(data, indices, num_segments, gather_scatter_method="grouped", **kwargs):
    if gather_scatter_method == "grouped":
        x = xpu.grouped_scatter_sum(data, indices, table_size=num_segments)
    elif gather_scatter_method == "dense":
        # here, senders and receivers are dense matrices
        x = tf.matmul(tf.cast(indices, data.dtype), data, transpose_a=True)
    elif gather_scatter_method == "debug":
        unbatched_xs = [
            tf.math.unsorted_segment_sum(data[i], indices[i], num_segments=num_segments) for i in range(data.shape[0])
        ]
        x = tf.stack(unbatched_xs)
    else:
        raise ValueError(f"gather_scatter method {gather_scatter_method} is invalid.")
    return x


def _scatter_max(data, indices, num_segments, gather_scatter_method="grouped", backwards_mode="sum"):
    if gather_scatter_method == "grouped":
        x = xpu.grouped_scatter_max(data, indices, table_size=num_segments, backwards_mode=backwards_mode)
    elif gather_scatter_method == "dense":
        raise NotImplementedError("No dense mode available for scatter_max")
    elif gather_scatter_method == "debug":
        unbatched_xs = [
            tf.math.unsorted_segment_max(data[i], indices[i], num_segments=num_segments) for i in range(data.shape[0])
        ]
        x = tf.stack(unbatched_xs)

        # ensure init values get masked out
        min_value = -65500 if x.dtype == tf.float16 else -3.402823e38
        x = tf.where(x <= min_value, tf.cast(0, x.dtype), x)

    else:
        raise ValueError(f"gather_scatter method {gather_scatter_method} is invalid.")
    return x


def _scatter_min(data, indices, num_segments, **kwargs):
    return -_scatter_max(-data, indices, num_segments, **kwargs)


def _scatter_count(data, indices, num_segments):
    dummy_nodes = tf.ones_like(indices, dtype=data.dtype)
    return xpu.grouped_scatter_sum(dummy_nodes[..., tf.newaxis], indices, table_size=num_segments)


def _scatter_mean(data, indices, num_segments, **kwargs):
    sum = _scatter_sum(data, indices, num_segments, **kwargs)
    counts = _scatter_count(data, indices, num_segments)
    return sum / tf.maximum(counts, tf.constant(1, counts.dtype))


def _scatter_var(data, indices, num_segments, stable_var=True, **kwargs):
    dtype_org = data.dtype
    if stable_var:
        counts = _scatter_count(data, indices, num_segments)
        counts = tf.maximum(counts, tf.constant(1, counts.dtype))
        data_mean = tf.cast(tf.reduce_mean(data, axis=1, keepdims=True), dtype=tf.float32)
        data = tf.cast(data, dtype=tf.float32)
        var = _scatter_sum((data - data_mean) * (data - data_mean), indices, num_segments, **kwargs) / tf.cast(
            counts, dtype=tf.float32
        )
    else:
        mean = tf.cast(_scatter_mean(data, indices, num_segments, **kwargs), dtype=tf.float32)
        data = tf.cast(data, dtype=tf.float32)
        mean_sqr = _scatter_mean(data * data, indices, num_segments, **kwargs)
        var = mean_sqr - mean * mean
    return tf.cast(tf.keras.activations.relu(var), dtype=dtype_org)


def _scatter_mean_and_var(data, indices, num_segments, stable_var=True, **kwargs):
    dtype_org = data.dtype
    sum = _scatter_sum(data, indices, num_segments, **kwargs)
    counts = _scatter_count(data, indices, num_segments)
    mean = sum / tf.maximum(counts, tf.constant(1, counts.dtype))
    # cast mean to float32 if the second memthod of calculating var is used
    mean = tf.cast(mean, dtype=tf.float32)
    if stable_var:
        data_mean = tf.cast(tf.reduce_mean(data, axis=1, keepdims=True), tf.float32)
        data = tf.cast(data, dtype=tf.float32)
        var = _scatter_sum((data - data_mean) * (data - data_mean), indices, num_segments, **kwargs) / tf.cast(
            counts, dtype=tf.float32
        )
    else:
        data = tf.cast(data, dtype=tf.float32)
        mean_sqr = _scatter_mean(data * data, indices, num_segments, **kwargs)
        var = mean_sqr - mean * mean
    return tf.cast(tf.concat([mean, tf.keras.activations.relu(var)], axis=-1), dtype=dtype_org)


def _scatter_std(data, indices, num_segments, **kwargs):
    var = _scatter_var(data, indices, num_segments, stable_var=True, **kwargs)
    return tf.math.rsqrt(tf.keras.activations.relu(var) + 1e-07)


def _scatter_sqrtN(data, indices, num_segments, **kwargs):
    sum = _scatter_sum(data, indices, num_segments, **kwargs)
    counts = _scatter_count(data, indices, num_segments)
    counts = tf.maximum(counts, tf.constant(1, counts.dtype))
    return sum * tf.math.rsqrt(counts)


def _scatter_softmax(data, indices, num_segments, beta=1.0, stable=True, **kwargs):
    data_t = data - tf.reduce_max(data) if stable else data
    data_t = tf.cast(data_t, tf.float32)
    beta = tf.cast(beta, data_t.dtype)
    denom = _scatter_sum(tf.exp(beta * data_t), indices, num_segments, **kwargs)
    denom = _gather(denom, indices)
    denom = tf.where(denom != 0, denom, tf.constant(1, denom.dtype))
    number = tf.exp(beta * data_t)
    rat = number / denom
    out = _scatter_sum(tf.cast(data, tf.float32) * rat, indices, num_segments, **kwargs)
    return tf.cast(out, data.dtype)
