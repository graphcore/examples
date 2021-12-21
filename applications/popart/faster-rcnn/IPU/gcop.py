# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""This file contain the utils class that can
improve programming efficiency on IPU by popart.
"""
import builtins
import numpy as np
import basic_func as bF
import combined_func as cF
import customized_ops as cOps
import warnings
from basic_func import safe_mode_on, safe_mode_off, set_builder, set_options, set_memory_proportion, load_model, set_seed, set_load_strict, load_strict
from initializers import zeros_initializer, ones_initializer, random_normal_initializer, variance_scaling_initializer, constant_initializer
from easydict import EasyDict as edict
import gc_session

layers = edict()
math = edict()
random = edict()
nn = edict()
image = edict()

# FLOAT|FLOAT16|INT8|INT16|INT32|UINT8|UINT16|UINT32|BOOL
float32, float16, int32, int16, int8, uint32, uint16, uint8, bool, = 'FLOAT', 'FLOAT16', 'INT32', 'INT16', 'INT8', 'UINT32', 'UINT16', 'UINT8', 'BOOL'
int64, uint64 = 'INT64', 'UINT64'

device = gc_session.DeviceScope


class DTYPES:
    def __init__(self, ):
        self.float32 = float32
        self.float16 = float16
        self.int32 = int32
        self.int16 = int16
        self.int8 = int8
        self.uint32 = uint32
        self.int64 = int64
        self.uint64 = uint64
        self.uint16 = uint16
        self.uint8 = uint8
        self.bool = bool


dtypes = DTYPES()

mappin_tf2npy = {
    float32: np.float32,
    float16: np.float16,
    int32: np.int32,
    int16: np.int16,
    int8: np.int8,
    uint32: np.uint32,
    uint16: np.uint16,
    uint8: np.uint8,
    bool: np.bool
}


def trainable_variables(scope=None):
    if scope is not None:
        raise NotImplementedError
    return bF.get_all_trainable_weights()


def name_scope(scope_name):
    return bF.name_scope(scope_name)


def variable_scope(scope_name, reuse=None):
    assert reuse is None  # TODO implement reuse
    return name_scope(scope_name)


def placeholder(shape=None, dtype=dtypes.float32, name=None):
    assert shape is not None  # TODO compatible with shape=None
    if name is None:
        name = ''
    return bF.add_input_tensor(dtype, shape, debugContext=name)


def abs(t, name=''):
    return bF.abs(t, debugContext=name)


def shape(input, out_type=dtypes.int32, name=''):
    input_shape = input.pureShape
    input_shape_np = np.array(input_shape, dtype=mappin_tf2npy[out_type])
    input_shape_tensor = bF.shapeConstant(input_shape_np, debugContext=name)
    return input_shape_tensor


def constant(value, dtype=None, shape=None, name='Const'):
    # value: A constant value (or list) of output type dtype.
    # dtype: check "DTYPES()"
    if isinstance(value, np.ndarray):
        value_np = value
    elif type(value) in [list, int, float]:
        value_np = np.array(value)
    else:
        raise NotImplementedError

    if dtype is not None:
        value_np = value_np.astype(mappin_tf2npy[dtype])

    if shape is not None:
        value_np = value_np.reshape(shape)
    result = bF.constant(value_np, debugContext=name)
    return result


def pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=''):
    mode = mode.lower()
    paddings = np.array(paddings).transpose([1, 0]).tolist()
    return bF.pad(tensor,
                  paddings,
                  mode=mode,
                  constant_value=constant_values,
                  debugContext=name)


def greater(x, y, name=''):
    x = bF.to_tensor(x)
    y = bF.to_tensor(y)
    return bF.greater(x, y, debugContext=name)


def less(x, y, name=''):
    x = bF.to_tensor(x)
    y = bF.to_tensor(y)
    return bF.less(x, y, debugContext=name)


def range(input_1, input_2=None, delta=1, dtype=dtypes.int32, name='range'):
    if input_2 is None:
        start, limit = 0, input_1
    else:
        start, limit = input_1, input_2
    return constant(np.arange(start=start, stop=limit,
                              step=delta).astype(mappin_tf2npy[dtype]),
                    name=name)


def meshgrid(*args, **kwargs):
    for arg in args:
        assert isinstance(
            arg, bF.ConstantTensor), 'not implement non-constant tensor input'
    args = [arg.data for arg in args]
    results = np.meshgrid(*args, **kwargs)
    results = [constant(result) for result in results]
    return results


def exp(x, name=''):
    return bF.exp(x, debugContext=name)


def log(x, name=''):
    return bF.log(x, debugContext=name)


def maximum(x, y, name=""):
    x = bF.to_tensor(x)
    y = bF.to_tensor(y)
    return bF.max([x, y], debugContext=name)


def multiply(x, y, name=''):
    return x * y


def minimum(x, y, name=""):
    x = bF.to_tensor(x)
    y = bF.to_tensor(y)
    return bF.min([x, y], debugContext=name)


def reduce_mean(input_tensor,
                axis=None,
                keepdims=0,
                name='',
                reduction_indices=None,
                keep_dims=None):
    assert reduction_indices is None and keep_dims is None, 'they are deprecated'
    if axis is None:
        axis = list(builtins.range(input_tensor.shape.ndims))
    return bF.reduceMean(input_tensor,
                         axes=axis,
                         keepdims=keepdims,
                         debugContext=name)


def reduce_max(input_tensor,
               axis=None,
               keepdims=0,
               name='',
               reduction_indices=None,
               keep_dims=None):
    assert reduction_indices is None and keep_dims is None, 'they are deprecated'
    if axis is None:
        axis = list(builtins.range(input_tensor.shape.ndims))
    return bF.reducemax(input_tensor,
                        axes=axis,
                        keepdims=keepdims,
                        debugContext=name)


def reduce_min(input_tensor,
               axis=None,
               keepdims=0,
               name='',
               reduction_indices=None,
               keep_dims=None):
    assert reduction_indices is None and keep_dims is None, 'they are deprecated'
    if axis is None:
        axis = list(builtins.range(input_tensor.shape.ndims))
    return bF.reducemin(input_tensor,
                        axes=axis,
                        keepdims=keepdims,
                        debugContext=name)


def reduce_sum(input_tensor,
               axis=None,
               keepdims=0,
               name='',
               reduction_indices=None,
               keep_dims=None):
    assert reduction_indices is None and keep_dims is None, 'they are deprecated'
    if axis is None:
        axis = list(builtins.range(input_tensor.shape.ndims))
    return bF.reduceSum(input_tensor,
                        axes=axis,
                        keepdims=keepdims,
                        debugContext=name)


def fill(dims, value, name=None):
    if isinstance(value, int):
        dtype = 'INT32'
    elif isinstance(value, float):
        dtype = 'FLOAT'
    else:
        raise NotImplementedError
    result = bF.ones(dims, dtype=dtype) * value
    return result


def cast(x, dtype, name=''):
    if dtype == 'float32':
        dtype = 'float'
    if type(x) in [int, float]:
        return constant(x, dtype=dtype, name=name)
    elif isinstance(x, bF.TTensor):
        return bF.cast(x, dtype, name)
    else:
        raise NotImplementedError('type not implemented')


def reshape(tensor, shape, name=''):
    return bF.reshape(tensor, shape, name)


def matmul(a, b, name=''):
    return bF.matmul(a, b, debugContext=name)


def one_hot(indices,
            depth,
            on_value=1,
            off_value=0,
            axis=None,
            dtype=None,
            name='name'):
    assert axis is None
    values = constant(np.asarray([off_value, on_value]).astype(np.int32),
                      name=name)
    result = bF.one_hot(indices, depth, values=values, debugContext=name)
    if dtype is not None:
        result = result.cast(dtype)
    return result


def argmax(input, axis=0, name='', dimension=None, output_type=dtypes.int64):
    result = bF.argmax(input, axis=axis, debugContext=name)
    if output_type != dtypes.int64:
        result = result.cast(output_type)
    return result


def transpose(a, perm=None, conjugate=False, name='transpose'):
    a_rank = len(a.pureShape)
    if perm is None:
        perm = [1, 0]
        assert a_rank == 2
    assert conjugate is False, 'not implement if conjugate is True'
    for dim in perm:
        assert dim < a_rank
    return bF.transpose(a, perm, debugContext=name)


def stack(values, axis=0, name='stack'):
    return cF.stack(values, dim=axis, debugContext=name)


def concat(values, axis, name='concat'):
    return cF.concat(values, axis, debugContext=name)


def clip_by_value(t, clip_value_min, clip_value_max, name=''):
    return bF.clip(t, clip_value_min, clip_value_max, debugContext=name)


def tile(input, multiples, name=''):
    multiples = bF.eliminate_constantTensor_from_list(multiples, int)
    multiples = constant(multiples)
    return bF.tile(input, multiples, debugContext=name)


def ceil(input, name=''):
    return bF.ceil(input, debugContext=name)


def sqrt(x, name=''):
    return bF.sqrt(x, debugContext=name)


def _cumsum(x, axis=0, exclusive=0, reverse=0, name=''):
    if isinstance(axis, int):
        axis = bF.to_tensor(axis)
    return bF.cumsum(x,
                     axis,
                     exclusive=exclusive,
                     reverse=reverse,
                     debugContext=name)


def _shuffle(value, seed=None, name=''):
    assert seed is None, "not implemented"
    return cF.random_shuffle(value, seed, name)


def squeeze(input, axis=None, name='', squeeze_dims=None):
    assert squeeze_dims is None, "squeeze_dims deprecated"
    if axis is None:
        axis = []
        for idx, dim in enumerate(input.pureShape):
            if dim == 1:
                axis.append(idx)
    return bF.squeeze(input, dims=axis, debugContext=name)


def expand_dims(input, axis, name=''):
    if isinstance(axis, list) or isinstance(axis, tuple):
        axis = bF.eliminate_constantTensor_from_list(axis, int)
    else:
        axis = [axis]
    return bF.unsqueeze(input, axis, debugContext=name)


def identity(input, name=''):
    return bF.identity(input, name)


def sigmoid(input, name=''):
    return bF.sigmoid(input, debugContext=name)


def slice(input_, begin, size, name=''):
    axes = list(builtins.range(len(begin)))
    input_shape = input_.pureShape
    assert len(begin) == len(size)
    ends = []
    for begin_ele, size_ele, shape_ele in zip(begin, size, input_shape):
        if size_ele == -1:
            ends.append(shape_ele)
        else:
            ends.append(begin_ele + size_ele)
    if isinstance(begin, list):
        begin = bF.to_tensor(begin)
    if isinstance(ends, list):
        ends = bF.to_tensor(ends)
    if isinstance(axes, list):
        axes = bF.to_tensor(axes)
    return bF.gc_slice(input_,
                       axes,
                       starts=begin,
                       ends=ends,
                       debugContext=name)


def _conv2d(inputs,
            filters,
            kernel_size=3,
            strides=1,
            padding='valid',
            use_bias=True,
            kernel_initializer=variance_scaling_initializer(),
            bias_initializer=None,
            data_format='channels_last',
            activation=None,
            fp16_on=False,
            name='',
            bias_constraint=None):
    if isinstance(strides, int):
        strides = [strides] * 2
    padding = padding.lower()
    assert data_format == 'channels_first', 'only channels_first implemented'
    assert padding in ('same', 'valid'), 'only same implemented'
    if not isinstance(kernel_size, int):
        kernel_size = kernel_size[0]
    c_in = inputs.pureShape[1]
    c_out = filters
    kernals_shape = [c_out, c_in, kernel_size, kernel_size]
    kernals_dtype = bF.mappin_gc2npy[inputs.dtype]

    if use_bias:
        bias_initializer = zeros_initializer(
        ) if bias_initializer is None else bias_initializer
        bias_data = bias_initializer([c_out], kernals_dtype)
    else:
        assert bias_initializer is None
        bias_data = None

    bias_training = False if bias_constraint == 'not update' else None
    result = cF.conv2d(inputs,
                       filters,
                       ksize=kernel_size,
                       bias=use_bias,
                       train=True,
                       strides=strides,
                       padding_mode=padding,
                       fp16_on=fp16_on,
                       filters_data=kernel_initializer(kernals_shape,
                                                       kernals_dtype),
                       bias_data=bias_data,
                       debugContext=name,
                       bias_training=bias_training)

    if activation is not None:
        result = activation(result)

    return result


def _dense(inputs,
           units,
           activation=None,
           use_bias=True,
           kernel_initializer=None,
           bias_initializer=zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           fp16_on=False,
           name='',
           reuse=None):
    assert kernel_regularizer is None, 'not implemented'
    assert bias_regularizer is None, 'not implemented'
    assert bias_constraint is None, 'not implemented'
    assert kernel_constraint is None, 'not implemented'
    assert activity_regularizer is None, 'not implemented'
    assert reuse is None, 'not implemented'

    kernals_shape = [inputs.pureShape[-1], units]
    kernals_dtype = bF.mappin_gc2npy[inputs.dtype]
    kernel_initializer = variance_scaling_initializer(
    ) if kernel_initializer is None else kernel_initializer
    weights_data = kernel_initializer(kernals_shape, kernals_dtype)
    if use_bias:
        bias_initializer = zeros_initializer(
        ) if bias_initializer is None else bias_initializer
        bias_data = bias_initializer([units], kernals_dtype)

    result = cF.fc(inputs,
                   num_units_out=units,
                   weights=weights_data,
                   bias=use_bias,
                   bias_data=bias_data,
                   train=trainable,
                   fp16_on=fp16_on,
                   debugContext=name)

    if activation is not None:
        result = activation(result)

    return result


def _equal(x, y, name=''):
    return bF.equal(x, y, debugContext=name)


def _logical_and(x, y, name=''):
    return bF.logical_and(x, y, debugContext=name)


def _logical_not(x, name=''):
    return bF.logical_not(x, debugContext=name)


def _logical_or(x, y, name=''):
    return bF.logical_or(x, y, debugContext=name)


def _greater_equal(x, y, name=''):
    return cF.greater_equal(x, y, debugContext=name)


def _less_equal(x, y, name=''):
    return cF.less_equal(x, y, debugContext=name)


def _softmax(logits, axis=-1, name=''):
    if logits.shape.ndims < 2:
        raise RuntimeError('dims should be greater or equal to 2')
    elif logits.shape.ndims > 2:
        orginal_shape = list(logits.pureShape)
        logits = logits.reshape([-1, orginal_shape[-1]])
        result = bF.softmax_2d(logits, axis=1, debugContext=name)
        return result.reshape(orginal_shape)
    else:
        return bF.softmax_2d(logits, axis=axis, debugContext=name)


def _bn(inputs,
        axis=-1,
        momentum=0.9,
        epsilon=1e-5,
        center=True,
        scale=True,
        training=True,
        trainable=True,
        fused=False,
        gamma_initializer=ones_initializer(),
        beta_initializer=zeros_initializer(),
        moving_mean_initializer=zeros_initializer(),
        moving_variance_initializer=ones_initializer(),
        fp16_on=False,
        name=''):
    assert axis == 1, 'axis should be declared explicitly, only channels first implemented'
    if name is None:
        name = ''
    if fused:
        warnings.warn('bn fused is off by default')
    assert center and scale
    train = trainable and training
    channels = [inputs.pureShape[1]]
    dtype = bF.mappin_gc2npy[inputs.dtype]
    weights = {
        'mean': moving_mean_initializer(channels, dtype),
        'var': moving_variance_initializer(channels, dtype),
        'scale': gamma_initializer(channels, dtype),
        'bias': beta_initializer(channels, dtype)
    }
    result = cF.batch_norm(inputs,
                           fp16_on=fp16_on,
                           train=train,
                           weights=weights,
                           momentum=momentum,
                           epsilon=epsilon,
                           debugPrefix=name)
    return result[0]


def _max_pooling2d(inputs,
                   pool_size=3,
                   strides=2,
                   padding='same',
                   data_format='channels_last'):
    strides = [strides] * 2
    pool_size = [pool_size] * 2
    padding = padding.lower()
    if padding == 'same':
        padding = [pool_size[0] // 2] * 4
    else:
        raise NotImplementedError
    assert data_format == 'channels_first'
    return bF.maxPooling(inputs,
                         strides=strides,
                         kernel_size=pool_size,
                         padding=padding)


def _relu(features, name=''):
    return bF.relu(features, name)


def _non_max_suppression_padded(boxes,
                                scores,
                                max_output_size,
                                iou_threshold=0.5,
                                score_threshold=None,
                                pad_to_max_output_size=False,
                                sorted_input=False,
                                canonicalized_coordinates=False,
                                tile_size=512,
                                name=''):
    assert pad_to_max_output_size, 'IPU version must pad the result'
    assert canonicalized_coordinates is False
    output_boxes, output_keep, num_valids = cOps.nms(
        scores,
        boxes,
        score_threshold=score_threshold,
        threshold=iou_threshold,
        numDetections=max_output_size,
        debugContext=name)
    return output_boxes, output_keep.squeeze(0), num_valids.squeeze(0)


def _sparse_softmax_cross_entropy_with_logits(labels, logits, name=None):
    assert labels.dtype in [dtypes.int32, dtypes.int64]
    assert logits.dtype in [dtypes.float16, dtypes.float32]
    assert logits.shape.ndims == 2
    assert labels.shape.ndims == 1
    softmax_logits = bF.softmax_2d(logits)
    return bF.nllloss(softmax_logits, labels, debugPrefix=name)


def _softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=''):
    assert axis == -1
    logits = bF.softmax_2d(logits)
    return bF.nllloss(logits, labels, debugPrefix=name)


def tmp_sorted_non_max_suppression_padded(scores, boxes, max_output_size,
                                          iou_threshold):
    assert scores.pureShape[0] == 1
    assert boxes.pureShape[0] == 1
    output_boxes, output_keep, num_valids = cOps.nms(
        scores, boxes, threshold=iou_threshold, numDetections=max_output_size)
    output_scores = cF.select_by_idx(scores[0], output_keep[0],
                                     dim=-1)  # TODO check the result
    return output_scores.unsqueeze(0), output_boxes


def tmp_multilevel_crop_and_resize(features,
                                   boxes,
                                   output_size=7,
                                   use_einsum_gather=False):
    num_rois = boxes.pureShape[1]
    results = []
    for level in features:
        current_feat = features[level]
        result = cOps.roi_align(
            current_feat, boxes,
            num_rois=num_rois)  # TODO check spatial scale is right
        results.append(result)
    result = concat([result.unsqueeze(-1) for result in results], -1)
    result = reduce_mean(result, axis=-1)
    return result


def where(condition, x=None, y=None, name=''):
    if x is None and y is None:
        raise NotImplementedError
    elif x is not None and y is not None:
        return bF.where(condition, x, y, debugContext=name)
    else:
        raise NotImplementedError


def ones_like(input, dtype=None, name=''):
    return bF.oneslike(input, dtype=dtype, debugContext=name)


def zeros_like(input, dtype=None, name=''):
    return bF.zeroslike(input, dtype=dtype, debugContext=name)


def zeros(shape, dtype=dtypes.float32, name=''):
    if isinstance(shape, int):
        shape = [shape]
    return bF.zeros(shape, dtype, name)


def ones(shape, dtype=dtypes.float32, name=''):
    if isinstance(shape, int):
        shape = [shape]
    return bF.ones(shape, dtype, name)


def gather(params, indices, validate_indices=None, axis=0, name=''):
    assert validate_indices is None, 'Deprecated, does nothing.'
    result = bF.gather(params, indices, dim=axis, debugContext=name)
    return result


def print_tensor(t):
    bF.print_tensor(t)


def _top_k(input, k=1, sorted=True, name=''):
    dim = input.shape.ndims - 1
    assert input.pureShape[-1] >= k
    k = bF.to_tensor(k, dtype='INT64')
    values, order = bF.topk(input,
                            k,
                            sorted=sorted,
                            dim=dim,
                            debugContext=name)
    return values, order


# implement nn's modules
nn.relu = _relu
nn.top_k = _top_k
nn.softmax = _softmax
nn.softmax_cross_entropy_with_logits = _softmax_cross_entropy_with_logits
nn.sparse_softmax_cross_entropy_with_logits = _sparse_softmax_cross_entropy_with_logits

# implement math's modules
math.equal = _equal
math.logical_and = _logical_and
math.logical_or = _logical_or
math.logical_not = _logical_not
math.greater_equal = _greater_equal
math.less_equal = _less_equal
math.cumsum = _cumsum

# implement random's modules
random.shuffle = _shuffle

# implement image's modules
image.non_max_suppression_padded = _non_max_suppression_padded

# implement layers's modules
layers.conv2d = _conv2d
layers.batch_normalization = _bn
layers.max_pooling2d = _max_pooling2d
layers.dense = _dense

# temporary ussage
DEBUG_TENSORS = {}


def enable_global_initializer(initializer):
    bF.enable_global_initializer(initializer)


def set_exclude_weights(keys):
    bF.set_exclude_weights(keys)


def collect_tensor(t, name):
    assert name not in DEBUG_TENSORS
    DEBUG_TENSORS[name] = t
