# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import numpy as np
import tensorflow.compat.v1 as tf
os.sys.path.append("../")  # dynamic_sparsity


class BertSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, opts, dtype):
        super(BertSchedule, self).__init__()
        self.dtype = dtype
        self.peak_lr = tf.constant(opts.peak_learning_rate, tf.float32)
        self.warmup_steps_i = tf.cast(opts.warmup_steps, tf.int32)
        self.warmup_steps_f = tf.cast(self.warmup_steps_i, tf.float32)
        self.decay_steps_i = tf.cast(opts.cooldown_steps, tf.int32)
        self.decay_steps_f = tf.cast(self.decay_steps_i, tf.float32)
        self.min_lr = opts.min_learning_rate
        self.decay_power = opts.decay_power

    def __call__(self, step):
        # Casting
        step_i = tf.cast(step, tf.int32)
        step_f = tf.cast(step_i, tf.float32)

        # Branching
        poly_learning_rate = tf.train.polynomial_decay(self.peak_lr, step_i - self.warmup_steps_i,
                                                       self.decay_steps_f, self.min_lr,
                                                       power=self.decay_power)

        def true_fn():
            return (step_f / self.warmup_steps_f) * self.peak_lr

        def false_fn():
            return poly_learning_rate

        return tf.cast(tf.cond(step_i < self.warmup_steps_i, true_fn, false_fn), self.dtype)


def build_optimizer(opt_name, opt_args=None):
    # Fetch the requested optimiser
    opt_cls = {
        'GradientDescent': tf.train.GradientDescentOptimizer,
        'Momentum': tf.train.MomentumOptimizer,
        'Adam': tf.train.AdamOptimizer
    }.get(opt_name)

    if opt_cls is None:
        raise ValueError(f'Unsupported optimizer {opt_name}')

    # Fetch default kwargs, accepting overrides from argparse
    opt_kws = {
        'GradientDescent': {},
        'Momentum': {
            'momentum': 0.0001,
            'use_nesterov': True
        },
        'Adam': {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-02
        }
    }.get(opt_name)
    if opt_args is not None:
        opt_kws.update(opt_args)

    return opt_cls, opt_kws


def make_histogram_proto(data, bins_count=None):
    # number of elements in the array
    elem_count = np.prod(data.shape)

    # Make sure the number of bins is defined and
    # doesn't exceed the nume of element
    if bins_count is None:
        bins_count = elem_count
    else:
        bins_count = np.min([bins_count, elem_count]).astype(np.int)

    # compute histogram using numpy
    occurrences, bin_edges = np.histogram(data, bins=bins_count)

    return tf.HistogramProto(min=data.min().astype(np.float),
                             max=data.min().astype(np.float),
                             num=elem_count.astype(np.int),
                             sum=np.sum(data).astype(np.float),
                             sum_squares=np.sum([datum * datum for datum in data]).astype(np.float),
                             bucket_limit=bin_edges[1:].tolist(),
                             bucket=occurrences.tolist())
