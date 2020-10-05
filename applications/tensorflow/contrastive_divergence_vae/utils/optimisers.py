# Copyright 2019 Graphcore Ltd.
# coding=utf-8
"""
Custom tensorflow optimisers
"""
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops, control_flow_ops


class VcdRMSPropOptimizer(tf.train.RMSPropOptimizer):
    """
    RMSProp used in VCD VAE paper. Two differences from standard RMSProp:
    1. different variable initialisations,
    2. update is learning_rate * g_t / (sqrt(mean_square) + epsilon) instead of
    learning_rate * g_t / sqrt(mean_square + epsilon)

    NOTE: Does not support centring, momentum, locking
    """

    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 epsilon=1.,
                 name="RMSPropVCD"):
        super(VcdRMSPropOptimizer, self).__init__(learning_rate,
                                                  momentum=0.,
                                                  decay=decay,
                                                  epsilon=epsilon,
                                                  use_locking=False,
                                                  centered=False,
                                                  name=name)

    def _create_slots(self, var_list):
        """Initialise variables to 0.01 (instead of 1. in standard tf RMSProp"""
        for v in var_list:
            if v.get_shape().is_fully_defined():
                init_rms = init_ops.constant_initializer(value=0.01)
            else:
                init_rms = array_ops.ones_like(v) * 0.01
            self._get_or_make_slot_with_initializer(v, init_rms, v.get_shape(),
                                                    v.dtype.base_dtype, "ms",
                                                    self._name)
            if self._centered:
                self._zeros_slot(v, "mg", self._name)
            self._zeros_slot(v, "momentum", self._name)
            self._get_or_make_slot_with_initializer(v, init_ops.zeros_initializer(),
                                                    tf.TensorShape(()),
                                                    v.dtype.base_dtype,
                                                    'is_first_iter',
                                                    self._name)

    def _apply_dense(self, grad, var):
        """
        mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
        mom = momentum * mom{t-1} + learning_rate * g_t / (sqrt(mean_square) + epsilon)
        delta = - mom
        :param grad:
        :param var:
        :return:
        """
        ms = self.get_slot(var, "ms")
        is_first_round = self.get_slot(var, "is_first_iter")
        decay = tf.cast(self._decay_tensor, var.dtype.base_dtype)
        lr = tf.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        eps = tf.cast(self._epsilon_tensor, var.dtype.base_dtype)
        do_decay = (1. - is_first_round)   # Don't decay on first iteration - set to grad_{t=0}^2
        ms_new = state_ops.assign(ms, decay * ms * do_decay + (1. - decay * do_decay) * grad * grad)

        with tf.control_dependencies([ms_new]):
            not_first_iter = state_ops.assign(is_first_round, 0.)
            var_update = state_ops.assign_sub(var, lr * grad / (eps + tf.sqrt(ms_new)))
            return control_flow_ops.group(*[var_update, not_first_iter])

    def _resource_apply_dense(self, grad, var):
        return self._apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        raise NotImplementedError('Sparse operations not supported')

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError('Sparse operations not supported')
