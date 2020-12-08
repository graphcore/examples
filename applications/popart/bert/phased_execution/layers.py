# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

"""Basic neural network layers."""
import ctypes
import logging
import os
from collections import namedtuple
from functools import reduce
from typing import List, Optional

import numpy as np

from phased_execution.nn import Block, Parameter, Scope


__all__ = ["Embedding", "Add", "Norm", "Dropout", "Dense"]

logger = logging.getLogger(__name__)


class Embedding(Block):
    def __init__(self,
                 scope: Scope,
                 input_dim: int,
                 output_dim: int,
                 dtype: np.dtype = np.float16,
                 weight_initializer: np.array = None,
                 custom: bool = False,
                 detach: bool = False,
                 weight_transposed: bool = False,
                 **kwargs):
        """Turns non-negative integers (indexes/tokens) into dense vectors
        of fixed size.

        Args:
            input_dim (int): Size of the vocabulary, i.e. maximum integer index + 1.
            output_dim (int): Dimension of the dense embedding.
            dtype (str, optional): Data type of output embeddings. Defaults to 'float16'.
            weight_initializer (np.array, optional): Initializer for the `embeddings` matrix. Defaults to None.

        Returns:
            str:  Output tensor of shape (x0, x1, ... xN-1, output_dim) where (x0, x1, .... xN-1) is shape of input tensor.
        """
        if weight_transposed:
            weight_shape = (output_dim, input_dim)
        else:
            weight_shape = (input_dim, output_dim)
        params = [
            Parameter(name="weight",
                      shape=weight_shape,
                      value=weight_initializer)
        ]
        self.weight_transposed = weight_transposed
        self.custom = custom
        self.detach = detach

        super(Embedding, self).__init__(params=params,
                                        scope=scope,
                                        dtype=dtype,
                                        **kwargs)
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, x: str):

        if self.weight_transposed:
            weight = self.builder.aiOnnx.transpose(
                [self.params[0].popart_tensor])
        else:
            weight = self.params[0].popart_tensor

        if self.custom:
            x = self.builder.customOp(
                opName="EmbeddingGather",
                opVersion=1,
                domain="ai.graphcore",
                attributes={},
                inputs=[weight, x],
            )[0]
        else:
            x = self.builder.aiOnnx.gather([weight, x])

        if self.detach:
            x = self.builder.customOp(opName="Detach",
                                      opVersion=1,
                                      domain="ai.graphcore",
                                      inputs=[x],
                                      attributes={"pass_through_creation":
                                                  1})[0]
        return x


class Add(Block):
    def __init__(self, scope, **kwargs):
        super(Add, self).__init__(params=[], scope=scope, **kwargs)

    def forward(self, x: List[str]):
        return reduce(
            lambda a, b: self.builder.aiOnnx.add([a, b],
                                                 debugPrefix=self.scope.name),
            x)


class Norm(Block):
    def __init__(self, scope, input_dim, epsilon, dtype, **kwargs):
        self.epsilon = epsilon
        params = [
            Parameter(name="Gamma",
                      shape=(input_dim, ),
                      value=np.ones(input_dim).astype(dtype)),
            Parameter(name="Beta",
                      shape=(input_dim, ),
                      value=np.zeros(input_dim).astype(dtype))
        ]
        super(Norm, self).__init__(params=params,
                                   scope=scope,
                                   dtype=dtype,
                                   **kwargs)

    def forward(self, input_x):
        gamma, beta = [param.popart_tensor for param in self.params]
        outs = self.builder.aiGraphcore.groupnormalization(
            [input_x, gamma, beta], 1, self.epsilon)
        return outs[0]


class Dropout(Block):
    def __init__(self, scope, dropout_prob, **kwargs):
        self.dropout_prob = dropout_prob
        super(Dropout, self).__init__(params=[], scope=scope, **kwargs)

    def forward(self, input_x: str):
        return self.builder.aiOnnx.dropout([input_x], 1, self.dropout_prob)[0]


class Transpose(Block):
    def _init_(self, scope, perm: None):
        self.perm = perm
        super(Transpose, self).__init__(params=[], scope=scope)

    def forward(self, input_x: str):
        return self.builder.aiOnnx.transpose([input_x], perm=self.perm)


Split = namedtuple('Split', ('dim', 'num_splits'))


class Dense(Block):
    def __init__(self,
                 scope: Scope,
                 input_dim: int,
                 output_dim: int,
                 split: Optional[Split] = None,
                 activation: 'str' = None,
                 available_memory_proportion: Optional[float] = None,
                 params: List[Parameter] = None,
                 alpha: float = 0.1,
                 use_default_memory_proportion: bool = True,
                 bias: bool = True,
                 **kwargs):
        scope.name = '/'.join([scope.name, 'Dense']) if scope.name else 'Dense'
        self.available_memory_proportion = available_memory_proportion
        self.split = split
        self.use_default_memory_proportion = use_default_memory_proportion
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim}
        self.bias = bias
        if params is None:
            params = [None, None]
        if self.bias:
            params = [
                Parameter(name='Weight', shape=[input_dim, output_dim], value=None)
                if not params[0] else params[0],
                Parameter(name='Bias', shape=[output_dim], value=np.zeros(output_dim).astype(kwargs['dtype']))
                if not params[1] else params[1]
            ]
        else:
            params = [Parameter(name='Weight', shape=[input_dim, output_dim], value=None)
                      if not params[0] else params[0]]
        super(Dense, self).__init__(params=params, scope=scope, **kwargs)
        if activation is None:
            self.activation = lambda x: x[0]
        else:
            activation = activation.lower()
            if activation == 'gelu':
                self.activation = self.builder.aiGraphcore.gelu
            elif activation == 'sgelu':
                self.activation = SimpliedGelu(scope=self.scope_provider.get_scope(
                                                    f'{scope.name}/sgelu', 'prev'),
                                               **kwargs)
            elif activation == 'relu':
                self.activation = self.builder.aiOnnx.relu
            elif activation == 'tanh':
                self.activation = self.builder.aiOnnx.tanh
            elif activation == 'lrelu':
                self.activation = LeakyRelu(scope=self.scope_provider.get_scope(
                                                  f'{scope.name}/lrelu', 'prev'),
                                            alpha=alpha,
                                            **kwargs)

    def forward(self, input_x: str):
        weight = self.params[0].popart_tensor
        x = self.builder.aiOnnx.matmul([input_x, weight])
        if self.split and self.split.num_splits:
            self.builder.setSerializeMatMul({x},
                                            self.split.dim,
                                            self.split.num_splits,
                                            keep_precision=True)
        if not self.use_default_memory_proportion:
            self.builder.setAvailableMemoryProportion(x, self.available_memory_proportion)
        if self.bias:
            bias = self.params[1].popart_tensor
            x = self.builder.aiOnnx.add([x, bias])
        return self.activation([x])


class SimpliedGelu(Block):
    def __init__(self, scope, dtype, **kwargs):
        self.dtype = dtype
        super().__init__(scope, [], dtype, **kwargs)

    def forward(self, input_x):
        """
            Simpler implementation of the GELU based on the sigmoid.
            Coming from the original Gelu paper (https://arxiv.org/abs/1606.08415).
        """
        input_x = input_x[0]
        scale = self.builder.aiOnnx.constant(
            np.asarray([1.702], dtype=self.dtype))
        result = self.builder.aiOnnx.mul([scale, input_x])
        result = self.builder.aiOnnx.sigmoid([result])
        result = self.builder.aiOnnx.mul([input_x, result])
        return result


class LeakyRelu(Block):
    def __init__(self, scope, alpha, **kwargs):
        self.alpha = alpha
        super().__init__(scope, [], **kwargs)

    def forward(self, input_x):
        """
            This function implements the leaky relu activation function.
            The mathematical function is:
            Leaky_Relu(x) = Relu(x) - alpha*Relu(-x)
        """
        input_x = input_x[0]
        alpha_t = self.builder.aiOnnx.constant(
            np.asarray([self.alpha], dtype=self._dtype))
        result_plus = self.builder.aiOnnx.relu([input_x])
        minus_x = self.builder.aiOnnx.neg([input_x])
        result_minus = self.builder.aiOnnx.relu([minus_x])
        result_minus = self.builder.aiOnnx.mul([alpha_t, result_minus])
        result = self.builder.aiOnnx.sub([result_plus, result_minus])
        return result
