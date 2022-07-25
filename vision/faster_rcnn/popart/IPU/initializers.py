# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import abc
import numpy as np
import time


# initializers
class Initializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __forward__(self, dst_shape, np_type):
        pass
        # return a np array

    def __call__(self, dst_shape, np_type=np.float32):
        assert isinstance(dst_shape, list)
        for num in dst_shape:
            assert isinstance(num, int)
        result = self.__forward__(dst_shape, np_type)
        assert isinstance(result, np.ndarray) or np.isscalar(result)
        return result


class zeros_initializer(Initializer):
    def __forward__(self, dst_shape, np_type):
        return np.zeros(dst_shape, dtype=np_type)


class ones_initializer(Initializer):
    def __forward__(self, dst_shape, np_type):
        return np.ones(dst_shape, dtype=np_type)


class random_normal_initializer(Initializer):
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        super().__init__()
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        if self.seed is not None:
            raise NotImplementedError('now use numpy seed')

    def __forward__(self, dst_shape, np_type):
        result = np.random.normal(loc=self.mean,
                                  scale=self.stddev,
                                  size=dst_shape).astype(np_type)
        return result


class variance_scaling_initializer(Initializer):
    def __init__(self,
                 scale=1.0,
                 mode='fan_in',
                 distribution='truncated_normal',
                 seed=None):
        super().__init__()
        if seed is not None:
            raise NotImplementedError('now use numpy seed')
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __forward__(self, dst_shape, np_type):
        c_out, c_in = dst_shape[0:2]
        if self.mode == 'fan_in':
            n = c_in
        elif self.mode == 'fan_out':
            n = c_out
        elif self.mode == 'fan_avg':
            n = (c_in + c_out) // 2
        else:
            raise NotImplementedError

        stddev = (self.scale / n)**0.5

        limit = (self.scale * 3 / n)**0.5
        if self.distribution == 'truncated_normal':
            result = np.random.normal(loc=0.0, scale=stddev,
                                      size=dst_shape).astype(np_type)
            result = np.clip(result, -limit, limit)
        elif self.distribution == 'untruncated_normal':
            result = np.random.normal(loc=0.0, scale=stddev,
                                      size=dst_shape).astype(np_type)
        elif self.distribution == 'uniform':
            result = np.random.uniform(-limit, limit, size=dst_shape)
        else:
            raise NotImplementedError
        return result


class constant_initializer(Initializer):
    def __init__(self, value=0):
        super().__init__()
        self.value = np.asarray(value)

    def __forward__(self, dst_shape, np_type):
        dst_size = 1
        for ele in dst_shape:
            dst_size *= ele
        if self.value.size > 1:
            assert self.value.size == dst_size
            result = self.value
        else:
            assert self.value.size == 1
            result = np.zeros(dst_shape, dtype=np_type) + self.value
        return result.reshape(dst_shape).astype(np_type)
