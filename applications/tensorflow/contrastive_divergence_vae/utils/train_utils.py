# Copyright 2019 Graphcore Ltd.
# coding=utf-8
import tensorflow as tf

from utils.optimisers import VcdRMSPropOptimizer


def vcd_lr_schedule(base_lr, current_step, total_steps, n_epoch, iter_timescale=15000, decay_factor=0.9):
    """
    Exponential LR decay: lr <- lr * 0.9 applied every 15000 iterations
    """
    n_timescales = tf.cast(current_step // iter_timescale, tf.float32)
    lr = base_lr * decay_factor ** n_timescales
    return lr


optimiser_configs = {
                     'vcd': [VcdRMSPropOptimizer,
                             {'decay': 0.9,
                              'epsilon': 1.,
                              'base_learning_rate': {'encoder': {'mean': 5e-4,
                                                                 'std': 2.5e-4},
                                                     'decoder': 5e-4},
                              'learning_rate_func': vcd_lr_schedule}]
                     }
