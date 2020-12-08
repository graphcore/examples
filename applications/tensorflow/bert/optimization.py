# coding=utf-8
# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been modified by Graphcore Ltd.


"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import, division, print_function

import logging
import operator
import re
from functools import reduce
from math import sqrt

import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
# pylint: enable=g-direct-tensorflow-import

from log import logger


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 loss_scaling=1.0,
                 exclude_from_weight_decay=[
                     "GroupNorm", "Group_Norm", "LayerNorm", "Layer_Norm", "bias"],
                 name="AdamWeightDecayOptimizer",
                 weights_dtype=tf.float16):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = tf.cast(learning_rate, dtype=weights_dtype)
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.loss_scaling = loss_scaling

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
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

            update = tf.cast(next_m / (tf.sqrt(next_v) + self.epsilon), param.dtype)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


class LAMBOptimizer(tf.train.Optimizer):
    """LAMB (Layer-wise Adaptive Moments optimizer for Batch training).

    This class has been adapted by Graphcore Ltd from Google Code at
    https://github.com/google-research/albert/blob/master/lamb_optimizer.py

    """
    # A new optimizer that includes correct L2 weight decay, adaptive
    # element-wise updating, and layer-wise justification. The LAMB optimizer
    # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
    # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
    # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)
    #

    def __init__(self,
                 learning_rate,
                 loss_scaling=1.0,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-4,
                 exclude_from_weight_decay=None,
                 exclude_from_layer_adaptation=None,
                 name="LAMBOptimizer",
                 high_precision=True,
                 use_nvlamb=False,
                 debiasing=True,
                 head_inner_dimension=64,
                 weight_clipping=0,
                 clipping_value=1.0):
        """Constructs a LAMBOptimizer."""
        super(LAMBOptimizer, self).__init__(False, name)

        # We assume that the weights are inputed in fp16.
        # High precision flag makes the last part of the optimizer being done in fp32
        # if it is false then the weight update is done in fp16

        self.beta_1 = tf.cast(beta_1, dtype=tf.float32)
        self.beta_2 = tf.cast(beta_2, dtype=tf.float32)
        self.loss_scaling = tf.cast(loss_scaling, dtype=tf.float32)
        self.epsilon = tf.cast(epsilon, dtype=tf.float16)
        logging.info("Setting Epsilon to {}".format(epsilon))

        self.high_precision = high_precision
        if self.high_precision:
            logging.info("Configured LAMB to use fp32 intermediate results")
            self.target_type = tf.float32
        else:
            logging.info("Configured LAMB to use fp16 intermediate results")
            self.target_type = tf.float16
        self.weight_decay_rate = weight_decay_rate
        self.weight_clip = weight_clipping
        if self.weight_clip:
            logging.info("Clipping the norm of the weights at {}".format(self.weight_clip))
        else:
            logging.info("Not clipping the norm of the weights.")
        self.learning_rate = tf.cast(learning_rate, dtype=self.target_type)

        self.exclude_from_weight_decay = exclude_from_weight_decay
        # If true use the NVLAM implimentaion found:
        self.use_nvlamb = use_nvlamb
        if self.use_nvlamb:
            logging.info("Using NVLAMB")

        # If true we debias the momenta (M and V) and if it is false we don't
        self.debiasing = debiasing
        if self.debiasing or self.use_nvlamb:
            logging.info("Using debiasing for M and V tensors")
        else:
            logging.info("Not using debiasing for M and V tensors")
        if self.use_nvlamb or self.debiasing:
            self.step = tf.get_variable('lamb_step_counter', dtype=tf.int32, initializer=[1], trainable=False)
        #  https://developer.nvidia.com/blog/pretraining-bert-with-layer-wise-adaptive-learning-rates/
        # -----

        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

        self.head_inner_dimension = head_inner_dimension
        self.clipping_value = tf.cast(clipping_value, dtype=tf.float32)

    def forward_transform(self, input_tensor):
        # Input tensor dimension is [H, 3H]
        # In the code from the modeling the transformation is
        # [H, 3H] -> [H, 3, H] -> [3, H^2]
        hidden_dim = input_tensor.shape.as_list()[0]
        reshaped_param = tf.reshape(input_tensor, [hidden_dim, 3, hidden_dim])
        reshaped_param = tf.transpose(reshaped_param, [1, 0, 2])
        reshaped_param = tf.reshape(reshaped_param, [3, hidden_dim*hidden_dim])
        return reshaped_param

    def backward_transform(self, input_tensor):
        # This function performs the opposite transformation as the previous one
        # [3, H^2] -> [3, H, H] -> [H, 3, H] -> [H, 3H]
        hidden_dim = int(sqrt(input_tensor.shape.as_list()[-1]))
        backward_transformed_tensor = tf.reshape(input_tensor, [3, hidden_dim, hidden_dim])
        backward_transformed_tensor = tf.transpose(backward_transformed_tensor, [1, 0, 2])
        backward_transformed_tensor = tf.reshape(backward_transformed_tensor, [hidden_dim, 3*hidden_dim])
        return backward_transformed_tensor

    def clipped_norm(self, gradients_list):
        # We compute the total norm of the gradient
        squared_gradients = [tf.reduce_sum(tf.square(tf.cast(g, dtype=tf.float32) /
                                                     self.loss_scaling)) for g in gradients_list]
        global_norm = tf.add_n(squared_gradients)
        global_norm = tf.sqrt(global_norm)
        clipped_global_norm = tf.maximum(global_norm, self.clipping_value)
        return clipped_global_norm

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []

        if self.use_nvlamb:
            global_norm = self.clipped_norm([g for g, v in grads_and_vars])

        # We reverse the order of the gradients and variables based on their sizes
        ordered_grad_and_vars = sorted(
            grads_and_vars, key=lambda x: reduce(operator.mul, x[0].shape, 1))

        for (grad, param) in ordered_grad_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            # Momentum
            m = tf.get_variable(
                name=f"{param_name}/lamb_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            # Velocity
            v = tf.get_variable(
                name=f"{param_name}/lamb_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # We convert the gradient to fp32 and we rescale it
            cast_grad = tf.cast(grad, dtype=tf.float32)
            cast_grad = cast_grad/self.loss_scaling

            if self.use_nvlamb:
                # We de normalize the gradients
                cast_grad = cast_grad*self.clipping_value/global_norm

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) +
                tf.multiply(1.0 - self.beta_1, cast_grad))
            next_v = (
                tf.multiply(self.beta_2, v) +
                tf.multiply(1.0 - self.beta_2,
                            tf.square(cast_grad)))

            # Beta scaling of momentum and velocity
            if self.debiasing:
                m_hat = next_m / (1.0 - tf.pow(self.beta_1, tf.cast(self.step, dtype=tf.float32)))  # x10
                v_hat = next_v / (1.0 - tf.pow(self.beta_2, tf.cast(self.step, dtype=tf.float32)))  # x1000
            else:
                m_hat = next_m
                v_hat = next_v

            update = m_hat / (tf.sqrt(tf.math.abs(v_hat)) +
                              tf.cast(self.epsilon, dtype=v_hat.dtype))

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += tf.cast(self.weight_decay_rate, dtype=update.dtype) * tf.cast(param, dtype=update.dtype)

            if 'qkv' in param_name:
                # We reshape the parameters
                reshaped_param = self.forward_transform(param)
                reshaped_update = self.forward_transform(update)
            else:
                reshaped_param = tf.reshape(param, [-1])
                reshaped_update = tf.reshape(update, [-1])

            # Norms are then computed in fp32
            w_norm = linalg_ops.norm(tf.cast(reshaped_param, dtype = tf.float32), ord=2, axis=-1)
            u_norm = linalg_ops.norm(reshaped_update, ord=2, axis=-1)

            reshaped_update = tf.cast(reshaped_update, dtype = self.target_type)

            if self.weight_clip:
                w_norm = tf.math.minimum(w_norm, tf.cast(
                    self.weight_clip, dtype=w_norm.dtype))

            # We set the ratio to 1 if either the w norm and the u norms are 0
            ratio = array_ops.where(math_ops.greater(w_norm, 0),
                                    array_ops.where(math_ops.greater(u_norm, 0), (tf.cast(
                                        w_norm, dtype=tf.float32) / u_norm), tf.constant(1.0, dtype=tf.float32, shape=w_norm.shape)),
                                    tf.constant(1.0, dtype=tf.float32, shape=w_norm.shape))

            # We reshape the ration in order to be broadcastable
            ratio = tf.reshape(ratio, shape=ratio.shape.as_list()+[1])
            # We combine the learning rate and the ratio at fp32 and then go back to fp16
            ratio = ratio * tf.cast(self.learning_rate, dtype = tf.float32)
            ratio = tf.cast(ratio, dtype=tf.float16)
            update_with_lr = ratio * reshaped_update
            # Backward transform to the same as param
            if 'qkv' in param_name:
                update_with_lr = self.backward_transform(update_with_lr)
            else:
                update_with_lr = tf.reshape(update_with_lr, shape=param.shape)
            update_with_lr = tf.cast(update_with_lr, dtype=param.dtype)

            next_param = param - update_with_lr

            # We add the update for the parameters and the biases
            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        # We add the update for the step
        if self.use_nvlamb or self.debiasing:
            assignments.extend(
                [self.step.assign(self.step+1)]
            )
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def get_optimizer(learning_rate, opts):
    """Configure and return the optimizer"""

    loss_scaling = opts["loss_scaling"]
    if opts['reduction_type'] == "mean":
        loss_scaling *= opts['pipeline_depth']
    scaled_learning_rate = learning_rate / loss_scaling

    if opts['optimiser'].lower() == 'sgd':
        logger.info("Using optimiser: sgd")
        optimizer = tf.train.GradientDescentOptimizer(scaled_learning_rate)

    elif opts['optimiser'].lower() == 'momentum':
        logger.info("Using optimiser: momentum")
        optimizer = tf.train.MomentumOptimizer(
            scaled_learning_rate, momentum=opts['momentum'], use_nesterov=False)

    elif opts['optimiser'].lower() == 'adamw':
        logger.info(f"Using optimiser: AdamWeightDecayOptimizer")
        optimizer = AdamWeightDecayOptimizer(
            learning_rate,
            beta_1=opts["optimiser_beta1"],
            beta_2=opts["optimiser_beta2"],
            epsilon=opts["optimiser_epsilon"],
            loss_scaling=loss_scaling)

    elif opts['optimiser'].lower() == 'lamb':
        logger.info("Using optimiser: LAMB")
        optimizer = LAMBOptimizer(
            learning_rate,
            loss_scaling=loss_scaling,
            beta_1=opts["optimiser_beta1"],
            beta_2=opts["optimiser_beta2"],
            weight_decay_rate=opts["weight_decay"],
            high_precision=opts["increase_optimiser_precision"],
            use_nvlamb=opts["use_nvlamb"],
            epsilon=opts["optimiser_epsilon"],
            debiasing=opts["use_debiasing"])

    else:
        raise ValueError(f"Optimizer {opts['optimiser']} not recognised")

    return optimizer
