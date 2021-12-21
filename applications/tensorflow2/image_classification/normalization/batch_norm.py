# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow_addons.optimizers import DecoupledWeightDecayExtension
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn, math_ops
from typing import Any, Optional, Mapping


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
        def moving_avg_updates(x, moving_m, moving_v):
            def bw(dx):
                return dx, moving_m - mean, moving_v - variance
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


def add_bn_moving_vars_updates_to_optimizer(optimizer_class, bn_momentum=0.99, loss_scaling=1):
    loss_scaling = loss_scaling or 1

    class BatchNormMovingAveragesUpdater(optimizer_class):

        def __init__(self, *args, **kwargs,):
            super(BatchNormMovingAveragesUpdater, self).__init__(*args, **kwargs)
            self._update_step = 1 - bn_momentum

        def _create_slots(self, var_list):
            non_moving_vars = [var for var in var_list if 'moving_' not in var.name]
            self.non_moving_var_names = {var.name for var in non_moving_vars}
            return super()._create_slots(non_moving_vars)

        def _resource_apply_dense(self, grad, var, apply_state):
            if var.name in self.non_moving_var_names:
                return super()._resource_apply_dense(grad, var, apply_state)
            else:
                return tf.raw_ops.ResourceApplyGradientDescent(
                    var=var.handle,
                    alpha=math_ops.cast(self._update_step, grad.dtype.base_dtype),
                    delta=grad * loss_scaling,
                    use_locking=self._use_locking)

        def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
            if issubclass(optimizer_class, DecoupledWeightDecayExtension):
                non_moving_vars = [var for var in var_list if 'moving_' not in var.name]
                return super().minimize(loss, var_list, grad_loss=grad_loss, name=name,
                                        decay_var_list=non_moving_vars, tape=tape)
            else:
                return super().minimize(loss, var_list, grad_loss=grad_loss, name=name, tape=tape)

    return BatchNormMovingAveragesUpdater
