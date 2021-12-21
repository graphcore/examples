# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import basic_func as bF
import numpy as np
from _globals import GLOBAL_V


def temporary_init_weights(data, name, train=True, fp16_on=False, load_global_first=True):
    local_func = bF.addInitializedInputTensor if train else bF.constant
    global_initializer = bF.get_global_initializer()
    local_name = bF.get_builder().getNameScope(name)
    if global_initializer.get(local_name, None) is not None and load_global_first:
        # overwrite data
        if local_name not in bF.get_exclude_weights():
            new_data = bF.get_global_initializer()[local_name]
            assert np.all(
                np.asarray(new_data.shape) == np.asarray(
                    data.shape)), '{}: {} vs {}'.format(
                        name, new_data.shape, data.shape)
            data = new_data
    else:
        assert not bF.load_strict(
        ), 'The global load_strict has been set, but the missing weight is detected: {}'.format(name)
    data = data.astype(np.float16 if fp16_on else np.float32)
    weights = local_func(data, name)
    if train:
        GLOBAL_V['all_trainable_weights'].append(weights)
    GLOBAL_V['all_weights'].append(weights)
    return weights


def stack(tensors, dim, debugContext=""):
    tensors = [t.unsqueeze(dim) for t in tensors]
    return concat(tensors, dim, debugContext=debugContext)


def concat(tensor_list, dim, debugContext=""):
    """names: list of tensor name"""
    ranks = [len(t.pureShape) for t in tensor_list]
    assert len(set(ranks)) == 1  # ranks of them should be same
    assert dim < ranks[0]  # dim should less than rank
    if ranks[0] == 0:
        tensor_list = [t.unsqueeze(0) for t in tensor_list]
        result = bF._concat(tensor_list, dim, debugContext).unsqueeze(0)
    else:
        result = bF._concat(tensor_list, dim, debugContext)
    return result


def conv2d(input,
           filters,
           ksize=3,
           bias=True,
           train=True,
           strides=[1, 1],
           pads=[1, 1, 1, 1],
           dilations=[1, 1],
           group=1,
           filters_data=None,
           bias_data=None,
           fp16_on=None,
           weights_fp16_on=None,
           padding_mode='same',
           debugContext='conv',
           bias_training=None):

    cast_flag, input, fp16_on = bF.deduce_half(input, fp16_on)

    batch, c_in, height, width = input.pureShape
    if debugContext != '':
        debugContext = debugContext + '/'
    weights_shape = [filters, c_in, ksize, ksize]
    if filters_data is not None:
        assert np.all(
            np.asarray(filters_data.shape) == np.asarray(weights_shape))
    else:
        filters_data = np.ones(weights_shape, bF.mappin_gc2npy[input.dtype])
    local_weights_fp16_on = fp16_on if weights_fp16_on is None else weights_fp16_on
    weights = temporary_init_weights(filters_data,
                                     debugContext + "weight",
                                     fp16_on=local_weights_fp16_on,
                                     train=train)
    if fp16_on and local_weights_fp16_on is False:
        weights = weights.cast('FLOAT16')

    # init bias
    bias_shape = [filters]
    if bias_data is not None:
        assert bias
        assert np.all(np.asarray(bias_data.shape) == np.asarray(bias_shape))
    else:
        if bias:
            bias_data = np.zeros(bias_shape, bF.mappin_gc2npy[input.dtype])
        else:
            bias_data = None
    if bias_data is not None:
        bias_training = train if bias_training is None else bias_training
        bias = temporary_init_weights(bias_data,
                                      debugContext + "bias",
                                      fp16_on=fp16_on,
                                      train=bias_training)
    else:
        bias = False

    if padding_mode == 'same':
        pads = [ksize // 2] * 4
    elif padding_mode == 'valid':
        pads = [0] * 4
    else:
        raise NotImplementedError

    result = bF._conv2d(input,
                        weights,
                        bias,
                        strides=strides,
                        pads=pads,
                        dilations=dilations,
                        group=group,
                        debugContext=debugContext)
    if cast_flag:
        result = result.cast(cast_flag)
    return result


def batch_norm(x,
               train=False,
               fp16_on=None,
               weights={
                   'mean': None,
                   'var': None,
                   'scale': None,
                   'bias': None
               },
               momentum=0.9,
               epsilon=1e-5,
               debugPrefix="bn"):

    cast_flag, x, fp16_on = bF.deduce_half(x, fp16_on)

    batch, c_in, height, width = x.pureShape

    dst_type = bF.mappin_gc2npy[x.dtype]
    mean = np.zeros(c_in).astype(
        dst_type) if weights['mean'] is None else weights['mean']
    var = np.ones(c_in).astype(
        dst_type) if weights['var'] is None else weights['var']
    scale = np.ones(c_in).astype(
        dst_type) if weights['scale'] is None else weights['scale']
    bias = np.zeros(c_in).astype(
        dst_type) if weights['bias'] is None else weights['bias']

    with bF.name_scope(debugPrefix):
        mean = temporary_init_weights(mean,
                                      "running_mean",
                                      train,
                                      fp16_on=fp16_on)
        var = temporary_init_weights(var,
                                     "running_var",
                                     train,
                                     fp16_on=fp16_on)
        scale = temporary_init_weights(scale, "weight", train, fp16_on=fp16_on)
        bias = temporary_init_weights(bias, "bias", train, fp16_on=fp16_on)
        if train:
            result = bF._batchNorm(x,
                                   scale,
                                   bias,
                                   mean,
                                   var,
                                   5 if train else 1,
                                   momentum=momentum,
                                   epsilon=epsilon,
                                   debugContext='')
        else:
            mean = mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            var = var.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            bias = bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            eps = np.asarray(1e-6).astype(
                np.float16 if fp16_on else np.float32)
            result = (x - mean) / bF.sqrt(var + eps) * scale + bias
        results = [result, mean, var, mean, var]

    if cast_flag:
        results = [result.cast(cast_flag) for result in results]

    return results


def select_by_idx(arr, unsigned_keeps, dim):
    depth = arr.pureShape[dim]
    one_hot_selector = bF.one_hot(unsigned_keeps, depth)
    one_hot_selector = bF.cast(one_hot_selector, arr.dtype)
    new_arr = bF.matmul(one_hot_selector, arr)
    return new_arr


def fc(x,
       num_units_out,
       weights=None,
       train=False,
       bias=True,
       bias_data=None,
       fp16_on=None,
       debugContext=''):

    cast_flag, x, fp16_on = bF.deduce_half(x, fp16_on)

    shape = x.pureShape
    num_units_in = shape[-1]
    weights_shape = [num_units_in, num_units_out]

    if debugContext != '':
        debugContext = debugContext + '/'
    # init weights
    if weights is not None:
        assert np.all(np.asarray(weights.shape) == np.asarray(weights_shape))
    else:
        weights = np.ones(weights_shape, bF.mappin_gc2npy[x.dtype])

    weights = temporary_init_weights(weights,
                                     debugContext + "weight",
                                     train,
                                     fp16_on=fp16_on)

    # init bias
    bias_shape = [num_units_out]
    if bias_data is not None:
        assert bias
        assert np.all(np.asarray(bias_data.shape) == np.asarray(bias_shape))
    else:
        if bias:
            bias_data = np.zeros(bias_shape, bF.mappin_gc2npy[x.dtype])
        else:
            bias_data = None
    if bias_data is not None:
        bias = temporary_init_weights(bias_data,
                                      debugContext + "bias",
                                      train,
                                      fp16_on=fp16_on)
    else:
        bias = False

    x = bF.matmul(x, weights, debugContext=debugContext)
    if bias:
        x = x + bias

    if cast_flag:
        x = x.cast(cast_flag)

    return x


def greater_equal(x, y, debugContext=''):
    _g = bF.greater(x, y)
    _e = bF.equal(x, y)
    result = bF.logical_or(_g, _e)
    return result


def less_equal(x, y, debugContext=''):
    _g = bF.greater(y, x)
    _e = bF.equal(x, y)
    result = bF.logical_or(_g, _e)
    return result


def random_shuffle(x, seed=None, debugPrefix=""):
    if seed is not None:
        raise RuntimeError(
            'random seed is globally set by session.setRandomSeed')
    with bF.name_scope(debugPrefix):
        x = bF.cast(x, 'FLOAT')
        seeds = bF.randomuniformlike(x, high=6.0, low=-6.0)
        flatten_seeds = bF.flatten(seeds)
        flatten_seeds_shape = flatten_seeds.pureShape
        _K = bF.constant(np.asarray([flatten_seeds_shape[0]]).astype(np.int64))
        _, shuffle_indices = bF.topk(flatten_seeds, _K, dim=0)
        flatten_x = bF.flatten(x)
        shuffle_indices = bF.cast(shuffle_indices, 'INT32')
        shuffled_flatten_x = bF.gather(
            flatten_x,
            shuffle_indices,
            dim=0,
        )
        x_shape = x.pureShape
        target_shape = bF.constant(np.asarray(x_shape).astype(np.int64))
        shuffled_x = bF.reshape(shuffled_flatten_x, target_shape)
    return shuffled_x
