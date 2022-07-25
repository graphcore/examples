# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

from tensorflow import distribute
from tensorflow.python.ipu import sharding
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ipu import ipu_multi_worker_strategy
from tensorflow.python.ipu.ops import cross_replica_ops
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
                 weight_decay_filter_fn = lambda x: False,
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
        kwargs['colocate_gradients_with_ops'] = True
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
            # don't rescale batch norm moving average statistics as they are not affected by loss scaling
            summed_grads_and_vars = [(grad, var) if 'batch_norm/moving_' in var.name else (grad / self._grad_scale, var)
                                     for grad, var in summed_grads_and_vars]
        ret = self._optimizer.apply_gradients(summed_grads_and_vars, global_step, name)
        if self._sharded:
            sharding.propagate_sharding(ops.get_default_graph())
        return ret

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self):
        return self._optimizer.variables()
