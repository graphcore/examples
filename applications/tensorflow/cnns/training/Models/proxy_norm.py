# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import tensorflow as tf
import numpy as np
from scipy.special import erfinv
from tensorflow.python.framework import ops
from tensorflow.python.training import training_ops


class recompute():
    def __init__(self, do_recompute=True):
        self._do_recompute = do_recompute

    @staticmethod
    def fake_data_dep(x, dep):
        # data dependency for dep by adding "zero" to x where "zero" is constructed such that it wont be optimised out
        dep = tf.reshape(dep, [-1])[0]
        dep1 = tf.constant(2 ** -24, dtype=dep.dtype) * dep
        dep2 = tf.constant(2 ** -24, dtype=dep.dtype) * dep
        return x + tf.stop_gradient(dep1 - dep2)

    def __call__(self, fn):
        @tf.custom_gradient
        def fw(*args):
            y = fn(*args)

            def bw(dy):
                args_with_dependency = [self.fake_data_dep(a, dy) for a in list(args)]
                y_recomp = fn(*args_with_dependency)
                return tf.gradients(y_recomp, args_with_dependency, grad_ys=dy)

            return y, bw

        return fw if self._do_recompute else fn


def get_rand(num_rand=200):
    rand = 2 * (np.arange(num_rand) + 0.5) / float(num_rand) - 1
    return np.sqrt(2) * erfinv(rand)


def proxy_initialiser(num_channels, activation, proxy_epsilon, dtype):
    with tf.init_scope():
        proxy_init = get_rand()
        proxy_init = np.maximum(proxy_init, 0.) if activation == tf.nn.relu else (
                proxy_init / (1 + np.exp(-proxy_init)))
        proxy_mean_init, inv_proxy_std_init = proxy_init.mean(), 1 / np.sqrt(proxy_init.var() + proxy_epsilon)
        proxy_init = [0., 1., proxy_mean_init, inv_proxy_std_init, 0., 0.]
        proxy_init = np.array(proxy_init).reshape((-1, 1, 1, 1))
        proxy_init = np.repeat(proxy_init, num_channels, axis=-1).astype(np.float32)
        return tf.cast(proxy_init, dtype=dtype)


def proxynorm_activation(x, activation=None, proxy_norm=True, proxy_epsilon=None,
                         proxy_recompute=True, delay_scale=False):
    with tf.name_scope('act'):
        if proxy_norm:
            num_channels = int(x.get_shape()[-1])
            proxy = tf.get_variable('proxy',
                                    dtype=x.dtype,
                                    initializer=proxy_initialiser(
                                        num_channels, activation, proxy_epsilon, x.dtype),
                                    aggregation=tf.VariableAggregation.SUM,
                                    trainable=True)
            beta, gamma, proxy_mean, inv_proxy_std = \
                (tf.cast(proxy[ind:ind + 1, :, :, :], x.dtype) for ind in range(4))

            @recompute(do_recompute=proxy_recompute)
            def proxyact_fw(x_in, _gamma, _beta, _proxy_mean, _inv_proxy_std):
                x = _gamma * x_in + _beta
                x = activation(x)
                # cast to float32 to ensure bwds reduce sum doesnt overflow
                x = tf.cast(x, proxy_mean.dtype)
                x = x - _proxy_mean
                x = tf.cast(x, x_in.dtype)
                if not delay_scale:
                    x = x * _inv_proxy_std
                return x

            x = proxyact_fw(x, gamma, beta, proxy_mean, inv_proxy_std)

            return (x, inv_proxy_std) if delay_scale else x
        else:
            return activation(x)


def make_pn_optimiser(optimiser_class,
                      proxy_filter_fn=None,
                      activation=None,
                      proxy_epsilon=0.03,
                      pipeline_splits=[],
                      dtype=tf.float16,
                      weight_decay=None,
                      ):
    class PNOptimiser(optimiser_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._proxy_filter_fn = proxy_filter_fn
            self._activation = activation
            self._dtype = dtype
            self._weight_decay = weight_decay
            self._proxy_epsilon = tf.cast(proxy_epsilon, self._dtype)
            self._rand = tf.constant(get_rand(), dtype=self._dtype)
            self._rand = tf.reshape(self._rand, shape=(1, 1, 1, 1, -1))
            self._pipeline_splits = [] if pipeline_splits is None else [s for s in pipeline_splits if type(s) == str]

        def proxy_fw(self, var, rand):
            rand = tf.cast(rand, self._dtype)
            _var = tf.cast(var, self._dtype)
            beta_prime, gamma_prime = _var[4:5, :, :, :], _var[5:6, :, :, :]
            beta_prime = tf.expand_dims(beta_prime, axis=4)
            gamma_prime = tf.expand_dims(gamma_prime, axis=4)
            rand = (1 + gamma_prime) * rand + beta_prime

            beta, gamma = _var[0:1, :, :, :], _var[1:2, :, :, :]
            beta = tf.expand_dims(beta, axis=4)
            gamma = tf.expand_dims(gamma, axis=4)
            proxy_x = gamma * rand + beta
            proxy_x = self._activation(proxy_x)
            proxy_x = tf.cast(proxy_x, tf.float32)
            proxy_mean, proxy_var = tf.nn.moments(proxy_x, axes=[4], keepdims=True)
            inv_proxy_std = tf.rsqrt(proxy_var + tf.cast(self._proxy_epsilon, tf.float32))
            proxy_mean, inv_proxy_std = tf.cast(proxy_mean, var.dtype), tf.cast(inv_proxy_std, var.dtype)
            proxy_mean = tf.reshape(proxy_mean, shape=(1, 1, 1, -1))
            inv_proxy_std = tf.reshape(inv_proxy_std, shape=(1, 1, 1, -1))
            updated_var = [var[0:2, :, :, :], proxy_mean, inv_proxy_std]
            updated_var += [var[4:6, :, :, :]]
            updated_var = tf.concat(updated_var, axis=0)
            return updated_var

        def proxy_bw(self, var, grad, rand):
            # manual recomputation and backprop of proxy variables
            updated_var = self.proxy_fw(var, rand)
            dot = tf.reduce_sum(updated_var[2:4, :, :, :] * tf.stop_gradient(grad[2:4, :, :, :]))
            grad = grad + tf.gradients(dot, var)[0]
            weight_decay = [0., 0., 0., 0., self._weight_decay, self._weight_decay]
            weight_decay = np.array(weight_decay, dtype=np.float32).reshape((-1, 1, 1, 1))
            weight_decay = tf.constant(weight_decay)
            weight_decay = tf.cast(weight_decay, var.dtype)
            grad = grad + weight_decay * var
            return grad

        def apply_proxy_gradients(self, proxy_grads_and_vars):
            lengths = [v.get_shape().as_list()[-1] for g, v in proxy_grads_and_vars]
            grads, vars = zip(*proxy_grads_and_vars)
            with ops.init_scope():
                self._create_slots(vars)
            all_grads = tf.concat(grads, axis=-1)
            all_vars = tf.concat(vars, axis=-1, name='Proxy_concat')

            rand = tf.constant(get_rand(), dtype=self._dtype)
            rand = tf.reshape(rand, shape=(1, 1, 1, 1, -1))

            all_grads = self.proxy_bw(all_vars, all_grads, rand)
            grads = tf.split(all_grads, lengths, axis=-1)
            updated_vars = []
            for grad, var in zip(grads, vars):
                updated_var = super()._apply_weight_update(grad, var)
                updated_vars.append(updated_var)
            updated_vars = tf.concat(updated_vars, axis=-1)

            # do proxy forward pass for the next step
            updated_vars = self.proxy_fw(updated_vars, rand)
            updated_vars = tf.split(updated_vars, lengths, axis=-1)

            grads_and_vars = [(v - up_v, v) for v, up_v in zip(vars, updated_vars)]

            return tf.group([training_ops.resource_apply_gradient_descent(
                var.handle, tf.constant(1.0, grad.dtype.base_dtype), grad,
                use_locking=self._use_locking) for grad, var in grads_and_vars])

        def apply_gradients(self, grads_and_vars, *args, **kwargs):
            if len(self._pipeline_splits) > 0 and self._pipeline_splits[0]:
                # select all proxy variables in the pipeline stage
                splits = self._pipeline_splits
                proxy_grads_and_vars = [[]]
                _grads_and_vars = []
                i = 0
                for grad, var in grads_and_vars:
                    if self._proxy_filter_fn(var.name):
                        if i < len(splits) and splits[i] in var.name:
                            i += 1
                            proxy_grads_and_vars += [[]]
                        proxy_grads_and_vars[-1] += [(grad, var)]
                    else:
                        _grads_and_vars.append((grad, var))
                assert i == len(splits)
            else:
                proxy_grads_and_vars = [
                    [(grad, var) for grad, var in grads_and_vars if self._proxy_filter_fn(var.name)]]
                _grads_and_vars = [
                    (grad, var) for grad, var in grads_and_vars if not self._proxy_filter_fn(var.name)]

            apply_op = super().apply_gradients(_grads_and_vars, *args, **kwargs)
            with tf.control_dependencies([apply_op]):
                return tf.group([self.apply_proxy_gradients(pg) for pg in proxy_grads_and_vars])

    return PNOptimiser
