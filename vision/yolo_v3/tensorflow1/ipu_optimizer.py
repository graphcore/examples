# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import logging
import operator
import re
from functools import reduce
from math import sqrt

import tensorflow as tf
from tensorflow import distribute
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.framework import ops
from tensorflow.python.ipu import internal_ops, ipu_multi_worker_strategy, sharding
from tensorflow.python.ipu.ops import cross_replica_ops
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops, init_ops, linalg_ops, math_ops, variable_scope
from tensorflow.python.training import optimizer


class IPUOptimizer(optimizer.Optimizer):
    def __init__(self,
                 optimizer,
                 sharded,
                 replicas,
                 gradient_accumulation_count,
                 pipelining=False,
                 grad_scale=1.0,
                 weight_decay=0.0,
                 weight_decay_filter_fn=lambda x: False,
                 var_list=None
                 ):
        super(IPUOptimizer, self).__init__(False, name="IPUOptimizer")
        self._optimizer = optimizer
        self._sharded = sharded
        self._replicas = replicas
        self._gradient_accumulation_count = gradient_accumulation_count
        self._pipelining = pipelining
        self._grad_scale = grad_scale
        self._weight_decay = weight_decay
        self._weight_decay_filter_fn = weight_decay_filter_fn
        self._var_list = var_list

    def add_WD(self, grads_and_vars):
        if self._weight_decay != 0.0:
            grads_and_vars = [
                (grad + (self._weight_decay * var), var) if self._weight_decay_filter_fn(var.name) else (grad, var)
                for grad, var in grads_and_vars]
        return grads_and_vars

    def compute_gradients(self, loss, var_list=None, **kwargs):
        if not var_list:
            var_list = self._var_list
        grads_and_vars = self._optimizer.compute_gradients(loss, var_list=var_list, **kwargs)
        if not self._pipelining:
            grads_and_vars = self.add_WD(grads_and_vars)
        if self._gradient_accumulation_count > 1:
            grads_and_vars = [(grad/self._gradient_accumulation_count, var)
                              for grad, var in grads_and_vars]
        if self._sharded:
            sharding.propagate_sharding(ops.get_default_graph())
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        summed_grads_and_vars = []
        for (grad, var) in grads_and_vars:
            if grad is None:
                summed_grads_and_vars.append((grad, var))
            else:
                with ops.colocate_with(grad):
                    # gradient accumulation
                    if self._gradient_accumulation_count > 1 and not self._pipelining:
                        grad = gen_poputil_ops.ipu_stateful_gradient_accumulate(grad,
                                                                                num_mini_batches=self._gradient_accumulation_count)

                    # replication
                    if self._replicas > 1:
                        grad = gen_poputil_ops.ipu_replication_normalise(cross_replica_ops.cross_replica_sum(grad))

                    # distribution with IPUMultiWorkerStrategy needs additional normalisation by the number of workers
                    if isinstance(distribute.get_strategy(), ipu_multi_worker_strategy.IPUMultiWorkerStrategy):
                        grad /= distribute.get_strategy().num_replicas_in_sync
                    grad = math_ops.cast(grad, var.dtype)
                    summed_grads_and_vars.append((grad, var))

        if self._pipelining:
            # can do weight decay here as apply_gradients is only called on last accumulation step
            summed_grads_and_vars = self.add_WD(summed_grads_and_vars)

        if self._grad_scale != 1.0:
            summed_grads_and_vars = [(grad / self._grad_scale, var) for grad, var in summed_grads_and_vars]
        ret = self._optimizer.apply_gradients(summed_grads_and_vars, global_step, name)
        if self._sharded:
            sharding.propagate_sharding(ops.get_default_graph())
        return ret

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self):
        return self._optimizer.variables()


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 use_moving_avg=False,
                 moving_avg_decay=0.995,
                 loss_scaling=1.0,
                 weight_decay_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=[
                     "GroupNorm", "Group_Norm", "LayerNorm", "Layer_Norm", "bias", "batch_norm"],
                 name="AdamWeightDecayOptimizer",
                 debiasing=False,
                 weights_dtype=tf.float16,
                 darknet_gn=False,
                 upsample_gn=False):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = tf.cast(learning_rate, dtype=weights_dtype)
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.loss_scaling = loss_scaling
        self.use_moving_avg = use_moving_avg
        self.moving_avg_decay = moving_avg_decay
        self.debiasing = debiasing
        self.darknet_gn = darknet_gn
        self.upsample_gn = upsample_gn

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        if self.debiasing:
            if global_step is None:
                local_step = variable_scope.get_variable(
                    name="local_step",
                    shape=[],
                    dtype=tf.float32,
                    initializer=init_ops.zeros_initializer(),
                    trainable=False)
                update_local_step = local_step.assign_add(1)
            else:
                update_local_step = global_step
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = _get_variable_name(param.name)
            # we use fp32 for calculating update
            grad_fp32 = tf.cast(grad, tf.float32)
            # We divide the gradient by the loss scaling
            grad_fp32 = grad_fp32/self.loss_scaling

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad_fp32))

            next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad_fp32)))
            if self.debiasing:
                next_m_debiase = next_m/(1.0 - tf.pow(self.beta_1, update_local_step))
                next_v_debiase = next_v/(1.0 - tf.pow(self.beta_2, update_local_step))
            else:
                next_m_debiase = next_m
                next_v_debiase = next_v

            update = tf.cast(
                next_m_debiase / (tf.sqrt(next_v_debiase) + self.epsilon), param.dtype)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if _do_use_weight_decay(param_name, self.weight_decay_rate, self.exclude_from_weight_decay):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            if distribute_ctx.has_strategy():
                # Handle DistributionStrategy case.
                if distribute_ctx.in_cross_replica_context():
                    raise RuntimeError("Use `_distributed_apply()` instead of "
                                       "`apply_gradients()` in a cross-replica context.")

                assign_params = distribute_ctx.get_replica_context().merge_call(
                    assign_vars, args=((param, m, v), (next_param, next_m, next_v)))
            else:
                assign_params = [
                    param.assign(next_param),
                    m.assign(next_m),
                    v.assign(next_v)]
            assignments.extend(assign_params)

            if _need_centering(param_name, self.darknet_gn, self.upsample_gn):
                with tf.control_dependencies(assign_params):
                    param_identity = tf.identity(param)
                centering_op = _centering_weights(param, param_identity)
                assignments.append(centering_op)

        if self.use_moving_avg:
            # using tf.train.ExponentialMovingAverage will make compiler produce many executables
            # and the program will run "load executable" for many times
            # so we write our own moving average
            # will use tf.train.ExponentialMovingAverage after we fix this
            assignments.extend(_create_moving_avg(grads_and_vars, self.moving_avg_decay))
        return tf.group(*assignments, name=name)


class MomentumOptimizer(tf.train.Optimizer):
    """
    Given different decay of learning rate and momentum to different weights at different pipeline stages.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 use_moving_avg=False,
                 moving_avg_decay=0.995,
                 loss_scaling=1.0,
                 weight_decay_rate=0.00,
                 dtype=tf.float16,
                 exclude_from_weight_decay=[
                     "GroupNorm", "Group_Norm", "LayerNorm", "Layer_Norm", "bias", "batch_norm"],
                 darknet_gn=False,
                 upsample_gn=False
                 ):
        super(MomentumOptimizer,
              self).__init__(False, name="MomentumOptimizer")
        self.learning_rate = learning_rate
        self.loss_scaling = loss_scaling
        self.momentum = tf.constant(momentum, dtype=dtype)
        self.weight_decay_rate = weight_decay_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.use_moving_avg = use_moving_avg
        self.moving_avg_decay = moving_avg_decay
        self.darknet_gn = darknet_gn
        self.upsample_gn = upsample_gn

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = _get_variable_name(param.name)
            m = tf.get_variable(
                name=param_name + "/momentum",
                shape=param.shape.as_list(),
                dtype=param.dtype,
                trainable=False,
                initializer=tf.zeros_initializer())

            next_m = self.momentum * m + grad

            update = next_m

            # update is scaled by loss_scaling
            # so we need to restore it's scale
            update /= self.loss_scaling
            if _do_use_weight_decay(param_name, self.weight_decay_rate, self.exclude_from_weight_decay):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            if distribute_ctx.has_strategy():
                # Handle DistributionStrategy case.
                if distribute_ctx.in_cross_replica_context():
                    raise RuntimeError("Use `_distributed_apply()` instead of "
                                       "`apply_gradients()` in a cross-replica context.")

                assign_params = distribute_ctx.get_replica_context().merge_call(
                    assign_vars, args=((param, m), (next_param, next_m)))
            else:
                assign_params = [param.assign(next_param), m.assign(next_m)]
            assignments.extend(assign_params)


            if _need_centering(param_name, self.darknet_gn, self.upsample_gn):
                with tf.control_dependencies(assign_params):
                    param_identity = tf.identity(param)
                centering_op = _centering_weights(param, param_identity)
                assignments.append(centering_op)

        if self.use_moving_avg:
            assignments.extend(_create_moving_avg(grads_and_vars, self.moving_avg_decay))

        return tf.group(*assignments, name=name)


def _need_centering(param_name, darknet_gn, upsample_gn):
    need_centering = False
    if param_name.startswith("darknet") and param_name.endswith("weight") and darknet_gn:
        need_centering = True
    if param_name.startswith("conv") and param_name.endswith("weight") and upsample_gn:
        need_centering = True
    return need_centering


def _centering_weights(weight, weight_identity):
    # when using group norm
    # normalize weights variable may get better result
    centered_weight = weight_identity - tf.reduce_mean(weight_identity, axis=[0, 1, 2])
    weight_norm = linalg_ops.norm(
        tf.cast(tf.reshape(centered_weight, [-1, centered_weight.shape[-1]]), dtype=tf.float32),
        ord=2,
        axis=-2)
    normed_weight = centered_weight / tf.cast(weight_norm, dtype=weight.dtype)

    if distribute_ctx.has_strategy():
        # Handle DistributionStrategy case.
        if distribute_ctx.in_cross_replica_context():
            raise RuntimeError("Use `_distributed_apply()` instead of "
                               "`apply_gradients()` in a cross-replica context.")

        assign_op = distribute_ctx.get_replica_context().merge_call(
            assign_vars, args=(weight, normed_weight))[0]
    else:
        assign_op = weight.assign(normed_weight)
    return [assign_op]


def _create_moving_avg(grads_and_vars, moving_avg_decay):
    with tf.name_scope("define_weight_decay"):
        moving_vars = []
        values = []
        for grad, param in grads_and_vars:
            # TODO: upadte variable creation in evalution.py in the same way would be better
            param_moving_avg = tf.get_variable(
                name=_get_variable_name(param.name)+"/ExponentialMovingAverage",
                shape=param.shape.as_list(),
                dtype=param.dtype,
                trainable=False,
                initializer=tf.zeros_initializer())
            decay_rate = moving_avg_decay
            moving_vars.append(param_moving_avg)
            values.append(param_moving_avg*decay_rate+param*(1-decay_rate))

        if distribute_ctx.has_strategy():
            # Handle DistributionStrategy case.
            if distribute_ctx.in_cross_replica_context():
                raise RuntimeError("Use `_distributed_apply()` instead of "
                                   "`apply_gradients()` in a cross-replica context.")

            moving_avgs = distribute_ctx.get_replica_context().merge_call(
                assign_vars, args=(moving_vars, values))
        else:
            moving_avgs = [var.assign(value) for var, value in zip(moving_vars, values)]
    with tf.control_dependencies(moving_avgs):
        no_op = tf.no_op()
    return [no_op]


def _do_use_weight_decay(param_name, weight_decay_rate, exclude_from_weight_decay):
    """Whether to use L2 weight decay for `param_name`."""
    if not weight_decay_rate:
        return False
    if exclude_from_weight_decay:
        for r in exclude_from_weight_decay:
            if re.search(r, param_name) is not None:
                return False
    return True


def _get_variable_name(param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
        param_name = m.group(1)
    return param_name


def assign_vars(distribution, variables, values):
    ops = []
    for variable, value in zip(variables, values):
        ops.append(variable.assign(value))
    return ops
