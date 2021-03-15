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

import logging
import textwrap
from abc import abstractmethod
from copy import copy
from typing import Any, Iterable, List, Optional
from collections import defaultdict

import numpy as np
from scipy.stats import truncnorm

import popart
from phased_execution.scope_manager import Scope


def normal_init_data(dtype, shape, mean, std_dev):
    # Truncated random normal between 2 standard devations
    data = truncnorm.rvs(-2, 2, loc=mean, scale=std_dev, size=np.prod(shape))
    data = data.reshape(shape).astype(dtype)
    return data


logger = logging.getLogger(__name__)


class Parameter:
    def __init__(self,
                 name: str,
                 shape: tuple,
                 value: Optional[np.ndarray] = None,
                 popart_tensor: Optional[str] = None):
        self.name = name
        self.shape = shape
        self.value = value
        self.popart_tensor = popart_tensor
        self.vgid = None

    def __repr__(self):
        return f'{self.popart_tensor}: shape={self.shape}'


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


class Block:
    """Base class for popart layers."""

    def __init__(self,
                 scope: Scope,
                 params: List[Parameter] = None,
                 dtype: np.dtype = np.float16,
                 builder=None,
                 initializers=None,
                 scope_provider=None,
                 tensors=None):

        self.scope = scope
        self._params = params if params is not None else []
        self._dtype = dtype
        self.num_params = 0
        self.sub_blocks = []
        self._tensors = tensors if tensors is not None else defaultdict(set)
        if builder is None:
            raise ValueError('Builder must be passed in as kwarg.')
        if scope_provider is None:
            raise ValueError('ScopeProvider must be passed in as kwarg.')

        self._builder = builder
        self.scope_provider = scope_provider
        self.initializers = {} if initializers is None else initializers

        self._total_flops = None
        self._total_params = None

        self._summary_widths = None
        self._summary_column_margin = 2
        self._summary_scope_str = '| '

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Block):
            self.sub_blocks.append(value)

        elif isinstance(value, Iterable):
            for el in flatten(value):
                if isinstance(el, Block):
                    self.sub_blocks.append(el)

        super(Block, self).__setattr__(name, value)

    @property
    def tensors(self):
        for blk in self.sub_blocks:
            for key in blk.tensors.keys():
                self._tensors[key] = self._tensors[key].union(blk.tensors[key])
        return self._tensors

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, _):
        raise ValueError('Cannot set builder property outside of Block class.')

    @property
    def summary_headers(self):
        return ['Layer (type)', 'Phase', 'VGID', 'Param #']

    @property
    def summary_fields(self):
        name = self.scope.name
        cls_name = self.__class__.__name__
        pp_phase = self.scope.execution_phase if hasattr(
            self.scope, 'execution_phase') else None
        return [
            name + ' (' + cls_name + ')',
            pp_phase,
            self.scope.vgid,
            f'{self.count_params():,}'
        ]

    def _set_summary_widths(self, widths):
        self._summary_widths = widths
        for blk in self.sub_blocks:
            blk._set_summary_widths(widths)

    def summary_widths(self, depth=0):
        def lengths(l_str):
            return np.array(list(map(lambda f: len(str(f)), l_str)))
        if self._summary_widths is None:
            max_widths = lengths(self.summary_fields)
            # Include indentation
            max_widths[0] += len(self._summary_scope_str)*depth
            for blk in self.sub_blocks:
                max_widths = np.maximum(max_widths, blk.summary_widths(depth+1))
            # Set the max_width on all blocks
            if depth is 0:
                max_widths = np.maximum(max_widths, lengths(self.summary_headers))
                self._set_summary_widths(max_widths)
            else:
                self._summary_widths = max_widths
        return self._summary_widths

    def format_row(self, fields):
        def align(i):
            return '<' if i < 1 else '>'
        line = ' '.join([
            '{!s:'+align(i)+str(w+self._summary_column_margin)+'}'
            for i, w in enumerate(self.summary_widths())])
        return line.format(*fields)

    def __repr__(self):
        s = self.format_row(self.summary_fields)

        def indent(s):
            res = ""
            i = 0
            while i < len(s):
                res += s[i]
                i += 1
                if res[-1] is '\n':
                    res += self._summary_scope_str
                    rem = s[i:]
                    jump = rem.index(' '*len(self._summary_scope_str))
                    res += s[i:i+jump]
                    i += jump + len(self._summary_scope_str)
            return res

        if len(self.sub_blocks) > 1:
            blocks = [s] + [str(block) for block in self.sub_blocks]
            rows_str = '\n'.join(blocks)
            s = indent(rows_str)
        return s

    def summary(self):
        header_str = self.format_row(self.summary_headers)
        logger.info(header_str)
        logger.info('_'*len(header_str))
        for blk in str(self).split('\n'):
            logger.info(blk)
        totals = self.format_row(['', '', 'Total:', f'{self.count_params():,}'])
        logger.info('_'*len(header_str))
        logger.info(totals)

    def num_flops(self):
        raise NotImplementedError()

    def count_flops(self):
        if self._total_flops is None:
            try:
                self._total_flops = self.num_flops()
            except NotImplementedError:
                self._total_flops = 0
            for blk in self.sub_blocks:
                self._total_flops += blk.count_flops()
        return self._total_flops

    def count_params(self):
        if self._total_params is None:
            child_params = 0
            for item in self.sub_blocks:
                child_params += item.count_params()
            self._total_params = child_params + self.num_params
        return self._total_params

    def __create_params__(self):
        # Create params only for a single block - not recursive
        for param in self._params:
            if param.popart_tensor is None:
                data = self.initializers.get(f'{self.builder.getNameScope()}{param.name}', param.value)
                if isinstance(data, np.ndarray):
                    if data.dtype != self._dtype:
                        raise ValueError(
                            f"Type of {param.name} does not match value provided. \n"
                            f"Provided {data.dtype}. Required {self._dtype}")
                    if np.any(data.shape != np.array(param.shape)):
                        if np.all(data.T.shape == np.array(param.shape)):
                            data = data.T.copy()
                            logger.warning(
                                f"Initializer for {param.name} was provided transposed."
                            )
                        else:
                            raise RuntimeError(
                                f"Initializer {param.name} does not match shapes. \n"
                                f" Provided {data.shape}. Required {param.shape}")
                    param.value = data
                else:
                    param.value = normal_init_data(self._dtype,
                                                   param.shape,
                                                   mean=0,
                                                   std_dev=0.02)

                popart_tensor = self._builder.addInitializedInputTensor(
                    param.value, param.name)
                self.num_params += np.prod(param.value.shape)
                param.vgid = self.scope.vgid
                self.tensors[0].add(popart_tensor)

            else:
                if (param.vgid != self.scope.vgid):
                    raise RuntimeError(
                        'Shared parameter {} appears in different virtual graphs.' .format(
                            param.name))
                # Transpose shared weights
                if (self.builder.getTensorShape(param.popart_tensor) == reversed(param.shape)):
                    popart_tensor = self._builder.aiOnnx.transpose([param.popart_tensor])
                else:
                    popart_tensor = param.popart_tensor
                # Create copy of shared param so as to not modify
                # the param for the previous layer
                param = copy(param)

            param.popart_tensor = popart_tensor

    @property
    def params(self):
        """Return this Block's parameter dictionary, not recursive.

        Returns:
            Dict: List of parameters for this block.
        """
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def next_phase(self):
        return self.scope_provider.get_next_phase()

    def total_phases(self):
        return self.scope_provider.get_prev_phase() + 1

    def __call__(self, *args):
        """Calls forward. Only accepts positional arguments."""
        with self.scope_provider(self._builder, self.scope):
            self.__create_params__()
            return self.forward(*args)

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError
