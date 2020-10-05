# Copyright 2020 Graphcore Ltd.
import logging
import textwrap
from abc import abstractmethod
from copy import copy
from typing import Any, Iterable, List, Optional

import numpy as np
from scipy.stats import truncnorm

import popart
from pingpong.scope_manager import Scope


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
                 params: List[Parameter] = [],
                 dtype: np.dtype = np.float16,
                 builder=None,
                 initializers=None,
                 scope_provider=None):

        self.scope = scope
        self._params = params
        self._dtype = dtype
        self.num_params = 0
        self.total_params = None
        self.line_length = 98
        self.sub_blocks = []
        if builder is None:
            raise ValueError('Builder must be passed in as kwarg.')
        if scope_provider is None:
            raise ValueError('ScopeProvider must be passed in as kwarg.')
        self._builder = builder
        self.scope_provider = scope_provider
        self.initializers = {} if initializers is None else initializers

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Block):
            self.sub_blocks.append(value)

        elif isinstance(value, Iterable):
            for el in flatten(value):
                if isinstance(el, Block):
                    self.sub_blocks.append(el)

        super(Block, self).__setattr__(name, value)

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, _):
        raise ValueError('Cannot set builder property outside of Block class.')

    def format_row(self, fields, indent=False):
        positions = [int(self.line_length * p) for p in [.45, .7, .8, 1.]]
        line = ''
        if indent:
            fields[0] = '    ' + fields[0]
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        return line

    def __repr__(self):
        name = self.scope.name
        cls_name = self.__class__.__name__
        pp_phase = self.scope.execution_phase if hasattr(
            self.scope, 'execution_phase') else None
        fields = [
            name + ' (' + cls_name + ')', pp_phase, self.scope.vgid,
            f'{self.count_params():,}'.rjust(10)
        ]
        s = self.format_row(fields)

        if len(self.sub_blocks) > 1:
            modstr = "\n".join([
                "{block}".format(block=block.__repr__())
                for block in self.sub_blocks
            ])
            s = "\n".join([s, modstr])
        return s

    def summary(self):
        header = ['Layer (type)', 'Pingpong Phase', 'VGID', 'Param #']
        print(self.format_row(header))
        print('_' * self.line_length)
        total_params = 0
        for blks in self.sub_blocks:
            print(blks)
            total_params += blks.count_params()

        print('Total params: {:,}'.format(total_params))

    def count_params(self):
        if self.total_params is not None:
            return self.total_params
        child_params = 0
        for item in self.sub_blocks:
            child_params += item.count_params()
        self.total_params = child_params + self.num_params
        return self.total_params

    def __create_params__(self):
        # Create params only for a single block - not recursive
        for param in self._params:
            if param.popart_tensor is None:
                data = self.initializers.get(param.name, None)
                if isinstance(data, np.ndarray):
                    if data.dtype != self._dtype:
                        raise ValueError(
                            f"Type of {param.name} does not match value provided. \n"
                            f"Provided {data.dtype}. Required {self._dtype}")
                    if np.any(data.shape != np.array(param.shape)):
                        if np.all(data.T.shape == np.array(param.shape)):
                            data = data.T.copy()
                            logger.warn(
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
                                                   std_dev=1)

                popart_tensor = self._builder.addInitializedInputTensor(
                    param.value, param.name)
                self.num_params += np.prod(param.value.shape)
                param.vgid = self.scope.vgid

            else:
                if param.vgid != self.scope.vgid:
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
