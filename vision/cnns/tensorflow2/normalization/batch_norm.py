# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow_addons.optimizers import DecoupledWeightDecayExtension
from tensorflow.python.framework import tensor_shape
from keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn, math_ops
from typing import Any, Optional, Mapping
import warnings
import numpy as np

class BatchNormIPU(tf.keras.layers.Layer):

    def __init__(self,
                 axis: int = -1,
                 center: bool = True,
                 scale: bool = True,
                 epsilon: float = 1e-3,
                 gamma_initializer: Any = 'ones',
                 beta_initializer: Any = 'zeros',
                 moving_mean_initializer: Any = 'zeros',
                 moving_variance_initializer: Any = 'ones',
                 trainable: bool = True,
                 name: Optional[str] = None,
                 **kwargs):

        super(BatchNormIPU, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
        self.trainable = trainable

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank.')
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %s' % (self.axis,))
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % (self.axis,))

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Received input '
                                 'with shape %s. Axis value: %s' %
                                 (tuple(input_shape), self.axis))
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)
        param_shape = (list(axis_to_dim.values())[0],)

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = tf.keras.backend.constant(
                1.0, dtype=self._param_dtype, shape=param_shape)

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                trainable=True,
                experimental_autocast=False)
        else:
            self.beta = tf.keras.backend.constant(
                0.0, dtype=self._param_dtype, shape=param_shape)

        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_mean_initializer,
            trainable=True,
            experimental_autocast=False)

        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=param_shape,
            dtype=self._param_dtype,
            initializer=self.moving_variance_initializer,
            trainable=True,
            experimental_autocast=False)

        self.built = True

    @property
    def _param_dtype(self):
        return self.dtype or tf.dtypes.float32

    def _get_training_value(self, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed from
            # model.
            training = False
        return training

    def call(self, inputs, training=None):
        training = self._get_training_value(training)
        inputs = tf.cast(inputs, tf.keras.mixed_precision.global_policy().variable_dtype)

        if training:
            outputs, mean, variance = nn.fused_batch_norm(
                inputs, self.gamma, self.beta, epsilon=self.epsilon)
        else:
            outputs, mean, variance = nn.fused_batch_norm(
                inputs,
                self.gamma,
                self.beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False)

        outputs = tf.cast(outputs, tf.keras.mixed_precision.global_policy().compute_dtype)

        @tf.custom_gradient
        def moving_avg_updates(x, moving_mean, moving_variance):
            def bw(dx):
                return dx, mean, variance
            return x, bw

        return moving_avg_updates(outputs, self.moving_mean, self.moving_variance)

    def get_config(self) -> Mapping[str, Any]:
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'trainable': self.trainable
        })
        return config


def add_bn_moving_vars_updates_to_optimizer(optimizer_class, model, batch_config, bn_momentum=0.99):

    class BatchNormMovingAveragesUpdater(optimizer_class):

        def __init__(self, *args, **kwargs,):
            super(BatchNormMovingAveragesUpdater, self).__init__(*args, **kwargs)

        def _create_slots(self, var_list):
            non_moving_vars = [var for var in var_list if 'moving_' not in var.name]
            self.non_moving_var_names = {var.name for var in non_moving_vars}
            return super()._create_slots(non_moving_vars)

        def _resource_apply_dense(self, grad, var, apply_state):
            bn_m = tf.cast(bn_momentum, grad.dtype)
            if var.name in self.non_moving_var_names:
                return super()._resource_apply_dense(grad, var, apply_state)
            else:
                return var.assign(
                    value=var * bn_m + grad*(1-bn_m),
                    use_locking=self._use_locking).op

        def _transform_gradients(self, grads_and_vars):

            # filter moving BN variables
            bn_moving_grads_and_vars = []
            non_moving_grads_and_vars = []
            for grad, var in grads_and_vars:
                if 'moving_' in var.name:
                    bn_moving_grads_and_vars.append((grad, var))
                else:
                    non_moving_grads_and_vars.append((grad, var))

            # only transform non moving grads
            non_moving_grads_and_vars = super()._transform_gradients(non_moving_grads_and_vars)

            return non_moving_grads_and_vars + bn_moving_grads_and_vars

        def _aggregate_gradients(self, grads_and_vars):

            moving_variance_grads_and_vars = []
            moving_mean_grads_and_vars = []
            remaining_grads_and_vars = []

            # filter moving BN mean and variance variables
            for grad, var in grads_and_vars:
                if 'moving_variance' in var.name:
                    moving_variance_grads_and_vars.append((grad, var))
                elif 'moving_mean' in var.name:
                    moving_mean_grads_and_vars.append((grad, var))
                else:
                    remaining_grads_and_vars.append((grad, var))

            assert len(moving_mean_grads_and_vars) == len(moving_mean_grads_and_vars), \
                'There should be an equal number of mean and variance variables to update'

            if len(moving_mean_grads_and_vars) > 0:
                agg_mov_mean_grad_and_vars = self.__aggregate_bn_moving_means(
                    moving_mean_grads_and_vars)

                agg_mov_variance_grads_and_vars = self.__agregate_bn_moving_variances(
                    moving_mean_grads_and_vars,
                    agg_mov_mean_grad_and_vars,
                    moving_variance_grads_and_vars)
            else:
                agg_mov_mean_grad_and_vars = []
                agg_mov_variance_grads_and_vars = []

            # normal aggregation applied to remaining gradients
            remaining_vars_aggregation = super()._aggregate_gradients(remaining_grads_and_vars)

            return agg_mov_mean_grad_and_vars + agg_mov_variance_grads_and_vars + remaining_vars_aggregation

        def __aggregate_bn_moving_means(self, mov_mean_grad_and_vars):
            grads, vars = zip(*mov_mean_grad_and_vars)
            grads = tf.distribute.get_replica_context().all_reduce(
                tf.distribute.ReduceOp.MEAN, grads)
            return list(zip(grads, vars))

        def __agregate_bn_moving_variances(self,
                                           mov_mean_grad_and_vars,
                                           agg_mean_grad_and_vars,
                                           mov_variance_grad_and_vars):

            dtype = mov_mean_grad_and_vars[0][0].dtype

            mov_mean_grads = [grad for grad, _ in mov_mean_grad_and_vars]
            agg_mean_grads = [grad for grad, _ in agg_mean_grad_and_vars]
            mov_variance_grads, mov_variance_vars = zip(*mov_variance_grad_and_vars)
            mov_variance_grads = list(mov_variance_grads)

            # variance is unbiased, we need to bias it
            for idx in range(len(mov_variance_grad_and_vars)):
                layer_name = '/'.join(mov_variance_vars[idx].name.split('/')[:-1])
                layer = model.get_layer(layer_name)
                elements_in_var = np.prod((batch_config.micro_batch_size, *layer.input.shape[1:-1]))
                mov_variance_grads[idx] *= tf.constant((elements_in_var - 1) / elements_in_var, dtype=dtype)

            # var_i = E[(mean_i - mean)^2 + var_i]
            per_replica_corrected_var = [tf.square(mov_mean - agg_mean) + mov_variance
                   for mov_mean, agg_mean, mov_variance in zip(mov_mean_grads, agg_mean_grads, mov_variance_grads)]
            mov_variance_grads = tf.distribute.get_replica_context().all_reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_corrected_var)

            # now we need to debias the previously biased variance
            for idx in range(len(mov_variance_grad_and_vars)):
                layer_name = '/'.join(mov_variance_vars[idx].name.split('/')[:-1])
                layer = model.get_layer(layer_name)
                elements_in_var = np.prod((batch_config.num_replicas, batch_config.micro_batch_size, *layer.input.shape[1:-1]))
                mov_variance_grads[idx] *= tf.constant(elements_in_var / (elements_in_var - 1), dtype=dtype)

            return list(zip(mov_variance_grads, mov_variance_vars))

        def _aggregate_moving_vars(self, bn_moving_grads_and_vars):

            grads, vars = zip(*bn_moving_grads_and_vars)

            all_reduced_vars = tf.distribute.get_replica_context().all_reduce(
                tf.distribute.ReduceOp.MEAN, grads)

            return list(zip(all_reduced_vars, vars))

        def apply_gradients(self,
                            grads_and_vars,
                            name=None,
                            experimental_aggregate_gradients=True,
                            **kwargs):

            if not experimental_aggregate_gradients:
                warnings.warn('IPU Batch Norm layer requires gradients to be aggregated to properly function. '
                              'Make sure to invoke optimizer._aggregate_gradients yourself.')

            return super().apply_gradients(grads_and_vars,
                                           name=name,
                                           experimental_aggregate_gradients=experimental_aggregate_gradients,
                                           **kwargs)

    return BatchNormMovingAveragesUpdater
