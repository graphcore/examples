# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Ross Girshick
"""This file contain the utils class that can
improve programming efficiency on IPU by popart.
"""
import builtins
import string
import numpy as np
import popart
from _globals import GLOBAL_V, set_batch, get_batch_size, get_anchor_return_type, train_mode_on, train_mode, safe_mode, safe_mode_on, safe_mode_off, get_builder, set_builder, set_seed, get_seed, set_options, get_options, set_device, get_device_type, get_ai_onnx_version, set_memory_proportion, get_memory_proportion, enable_global_initializer, get_global_initializer, get_exclude_weights, set_exclude_weights, get_all_trainable_weights, load_model, set_load_strict, load_strict

CONSTANT_COUNTER = [0]
TENSOR_NAMES = []


def unified_op_warpper(func):
    def warpper_func(*args, **kwargs):
        results = func(*args, **kwargs)
        return results

    return warpper_func


def name_scope(scope_name):
    return get_builder().nameScope(scope_name)


def add_input_tensor(dtype, shape, debugContext=''):
    input_t = get_builder().addInputTensor(popart.TensorInfo(dtype, shape),
                                           debugContext=debugContext)
    return TTensor(input_t)


def identity(input, debugContext=''):
    return TTensor(get_builder().aiOnnx.identity([input.getIpuIndex()],
                                                 debugContext=debugContext))


def pad(data, pads, mode='constant', constant_value=0, debugContext=''):
    constant_value = constant(constant_value).cast(data.dtype.upper())
    pads = to_tensor(pads).cast('INT64').flatten()
    result = get_builder().aiOnnx.pad(
        [data.getIpuIndex(),
         pads.getIpuIndex(),
         constant_value.getIpuIndex()],
        mode=mode,
        debugContext=debugContext)
    return TTensor(result)


def _conv2d(input,
            filter,
            bias=False,
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            dilations=[1, 1],
            group=1,
            debugContext=''):
    """Encapsulation of function get_builder().aiOnnx.conv!

    args:
        x:      input tensor
        ksize:  int,kernel size
        stride: int,stride of conv
        pads:   int, conv padding
        c_out:  int, output channel
        group:  int, conv group nums,default:1
    """
    args = [input.getIpuIndex(), filter.getIpuIndex()]
    if bias:
        args.append(bias.getIpuIndex())
    output = get_builder().aiOnnx.conv(args,
                                       strides=strides,
                                       pads=pads,
                                       dilations=dilations,
                                       group=group,
                                       debugContext=debugContext)
    if get_memory_proportion() is not None:
        get_builder().setAvailableMemoryProportion(output,
                                                   get_memory_proportion())
    return TTensor(output)


def relu(x, debugContext=""):
    """
    args:
        x:      input tensor
    """
    if isinstance(x, list):
        x = [ele.getIpuIndex() for ele in x]
    else:
        x = [x.getIpuIndex()]
    x = get_builder().aiOnnx.relu(x, debugContext=debugContext)
    return TTensor(x)


def maxPooling(x,
               strides=[2, 2],
               kernel_size=[2, 2],
               padding=[0, 0, 0, 0],
               dilations=[1, 1],
               ceil_mode=0,
               debugContext=""):
    """
    args:
        x:
        strides:     maxpool stride(output_size=input_size/strides)
        kernel_size: window's size that used to find max value
    """
    x = get_builder().aiOnnx.maxpool(args=[x.getIpuIndex()],
                                     num_outputs=1,
                                     kernel_shape=kernel_size,
                                     pads=padding,
                                     strides=strides,
                                     dilations=dilations,
                                     ceil_mode=ceil_mode,
                                     debugContext=debugContext)
    return TTensor(x[0])


def avgPooling(x,
               strides=2,
               kernel_size=2,
               padding=0,
               count_include_pad=0,
               debugContext=""):

    x = get_builder().aiOnnx.averagepool(
        [x.getIpuIndex()],
        kernel_shape=[kernel_size, kernel_size],
        count_include_pad=count_include_pad,
        pads=[padding] * 4,
        strides=[strides, strides],
        debugContext=debugContext)
    return TTensor(x)


def check_all_constant(tensors):
    for t in tensors:
        if isinstance(t, ConstantTensor) is False:
            return False
    return True


def resize(x,
           roi=None,
           scales=None,
           sizes=None,
           coordinate_transformation_mode='half_pixel',
           cubic_coeff_a=-0.75,
           exclude_outside=0,
           extrapolation_value=0.0,
           mode='nearest',
           nearest_mode='round_prefer_floor',
           debugContext=''):
    # TODO Check whether each parameter is correct
    # x:N-D tensor
    # roi: 1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X
    # scales: tensor(float),The scale array along each dimension.
    # sizes: tensor(int64),The size of the output tensor.
    # Only one of 'scales' and 'sizes' can be specified.
    assert None in [scales, sizes] and set([scales, sizes]) == 2
    if roi is None:
        assert coordinate_transformation_mode == 'tf_crop_and_resize'
    else:
        raise not NotImplementedError
    roi = constant(
        np.array([0, -1] * x.shape.ndims).astype(
            mappin_gc2npy[x.dtype])) if roi is None else roi
    scales = constant(np.array(
        [1.0] * x.shape.ndims).astype('FLOAT32')) if scales is None else scales
    sizes = constant(np.array(
        [1] * x.shape.ndims).astype('INT64')) if sizes is None else sizes
    inputs_list = [
        x.getIpuIndex(),
        roi.getIpuIndex(),
        scales.getIpuIndex(),
        sizes.getIpuIndex()
    ]
    inputs_dic = {
        'coordinate_transformation_mode': coordinate_transformation_mode,
        'cubic_coeff_a': cubic_coeff_a,
        'exclude_outside': exclude_outside,
        'extrapolation_value': extrapolation_value,
        'mode': mode,
        'nearest_mode': nearest_mode,
        'debugContext': debugContext
    }
    result = TTensor(get_builder().aiOnnx.resize(inputs_list, **inputs_dic))
    return result


def matmul(x, y, debugContext=""):
    if check_all_constant([x, y]):
        # degrade to np op
        result = np.matmul(x.data, y.data)
        return constant(result)
    else:
        assert x.dtype in ['FLOAT', "FLOAT16"]
        assert y.dtype in ['FLOAT', "FLOAT16"]
        return TTensor(get_builder().aiOnnx.matmul(
            [x.getIpuIndex(), y.getIpuIndex()], debugContext=debugContext))


def scalarTensor2int(t):
    if isinstance(t, ConstantTensor):
        return int(t.data)
    assert isinstance(t, int)
    return t


def reshape(source, target_shape, debugContext=""):
    """
    args:
        source : tensor name
        target_shape: list of int e.g.: [3,4,5,6]
    """
    if isinstance(target_shape, TTensor):
        target_shape = target_shape.data
    if isinstance(target_shape, np.ndarray):
        target_shape = target_shape.tolist()
    if isinstance(target_shape, list):
        target_shape = [scalarTensor2int(ele) for ele in target_shape]

    target_shape = constant(np.array(target_shape).astype(np.int64),
                            debugContext=debugContext)

    if check_all_constant([source, target_shape]):
        # degrade to np op
        result = source.data.reshape(target_shape.data)
        result = constant(result)
        return result
    else:
        return TTensor(get_builder().aiOnnx.reshape(
            [source.getIpuIndex(),
             target_shape.getIpuIndex()],
            debugContext=debugContext))


def softmax_2d(x, axis=1, debugContext=""):
    assert axis in [-1, 1]
    assert x.shape.ndims == 2
    x = get_builder().aiOnnx.softmax(
        [x.getIpuIndex()], axis=axis,
        debugContext=debugContext)
    return TTensor(x)


def _batchNorm(
    x,
    scale,
    biases,
    mean,
    var,
    num_outputs=1,
    momentum=0.9,
    epsilon=1e-5,
    debugContext="",
):
    results = get_builder().aiOnnx.batchnormalization(
        [
            x.getIpuIndex(),
            scale.getIpuIndex(),
            biases.getIpuIndex(),
            mean.getIpuIndex(),
            var.getIpuIndex()
        ],
        num_outputs=num_outputs,
        epsilon=epsilon,
        momentum=momentum,
        debugContext=debugContext)
    results = results[0] if num_outputs == 1 else results
    if isinstance(results, list):
        results = [TTensor(r) for r in results]
    else:
        results = [TTensor(results)]
    return results


def _concat(tensor_list, dim, debugContext=""):
    if check_all_constant(tensor_list):
        # degrade to np op
        np_arr_list = [t.data for t in tensor_list]
        result = np.concatenate(np_arr_list, axis=dim)
        return constant(result)
    return TTensor(get_builder().aiOnnx.concat(
        [tensor.getIpuIndex() for tensor in tensor_list],
        dim,
        debugContext=debugContext))


def sqrt(x, debugContext=""):
    result = get_builder().aiOnnx.sqrt([x.getIpuIndex()],
                                       debugContext=debugContext)
    return TTensor(result)


def sigmoid(tensor, debugContext=""):
    return TTensor(get_builder().aiOnnx.sigmoid([tensor.getIpuIndex()],
                                                debugContext=debugContext))


def transpose(x, dim_order, debugContext=""):
    """dim_order: list of int. eg:[0,2,3,1]"""
    if check_all_constant([x]):
        # degrade to np op
        result = np.transpose(x.data, dim_order)
        return constant(result)
    return TTensor(get_builder().aiOnnx.transpose([x.getIpuIndex()],
                                                  dim_order,
                                                  debugContext=debugContext))


def mean(x, debugContext=""):
    return TTensor(get_builder().aiOnnx.mean(x.getIpuIndex(),
                                             debugContext=debugContext))


def gc_slice(x, axes, starts, ends, debugContext=""):
    if check_all_constant([x, axes, starts, ends]):
        # degrade to np op
        x = x.data
        x_slices = []
        for start, end in zip(starts.data.tolist(), ends.data.tolist()):
            x_slices.append(slice(start, end))
        return constant(x[x_slices])
    else:
        x = get_builder().aiOnnx.slice([
            x.getIpuIndex(),
            starts.getIpuIndex(),
            ends.getIpuIndex(),
            axes.getIpuIndex()
        ],
            debugContext=debugContext)
    return TTensor(x)


def topk(x, k, sorted=True, dim=-1, debugContext=""):
    """
    args:
        k:      the count of return
        dim:    in which dim to sort and clip
    """
    if k.shape.ndims == 0:
        k = k.unsqueeze(0)
    else:
        assert k.shape.ndims == 1
    values, order = get_builder().aiOnnx.topk(
        [x.getIpuIndex(), k.getIpuIndex()],
        axis=dim,
        sorted=sorted,
        debugContext=debugContext)
    return TTensor(values), TTensor(order)


def constant(x, debugContext=''):
    if np.isscalar(x):
        return ConstantScalar(None, x)
    assert isinstance(x, np.ndarray)
    return ConstantTensor(None, x)


def shapeConstant(x, debugContext=''):
    if np.isscalar(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    return ShapeTensor(None, x)


def gather(x, indices, dim=0, debugContext=""):
    x = get_builder().aiOnnx.gather(
        [x.getIpuIndex(), indices.getIpuIndex()],
        axis=dim,
        debugContext=debugContext)
    return TTensor(x)


def addInitializedInputTensor(array, debugContext=""):
    """
    args:
        array: an numpy array that will be copy to IPU

        return:
            str: tensor name
    """

    name = get_builder().addInitializedInputTensor(array,
                                                   debugContext=debugContext)
    return TTensor(name)


def unsqueeze(x, dims, debugContext=""):
    """
    args:
        dim: list of int of which dim will delete
                eg:[3] or [1,3]
    """
    if check_all_constant([x]):
        # degrade to np op
        result = np.expand_dims(x.data, axis=dims)
        return constant(result)
    x = get_builder().aiOnnx.unsqueeze([x.getIpuIndex()],
                                       axes=dims,
                                       debugContext=debugContext)
    return TTensor(x)


def ceil(x, debugContext=""):
    result = get_builder().aiOnnx.ceil([x.getIpuIndex()],
                                       debugContext=debugContext)
    return TTensor(result)


def squeeze(x, dims, debugContext=""):
    if check_all_constant([x]):
        # degrade to np op
        x = x.data
        current_dim = float('inf')
        for dim in reversed(dims):
            assert current_dim > dim
            current_dim = dim
            x = x.squeeze(dim)
        return constant(x)
    if isinstance(dims, int):
        dims = [dims]
    for dim in dims:
        assert x.pureShape[dim] == 1
    x = get_builder().aiOnnx.squeeze([x.getIpuIndex()],
                                     axes=dims,
                                     debugContext=debugContext)
    return TTensor(x)


def exp(x, debugContext=""):

    return TTensor(get_builder().aiOnnx.exp([x.getIpuIndex()],
                                            debugContext=debugContext))


def printTensor(t):
    get_builder().aiGraphcore.printtensor([t.getIpuIndex()], print_gradient=0)


def mul(tensors, debugContext=""):
    if check_all_constant(tensors):
        # degrade to np op
        result = 1
        for t in tensors:
            result = t.data * result
        return constant(result, debugContext=debugContext)
    return TTensor(get_builder().aiOnnx.mul([t.getIpuIndex() for t in tensors],
                                            debugContext=debugContext))


def add(tensors, debugContext=""):
    if check_all_constant(tensors):
        # degrade to np op
        result = 0
        for t in tensors:
            result = result + t.data
        return constant(result, debugContext=debugContext)
    return TTensor(get_builder().aiOnnx.add([t.getIpuIndex() for t in tensors],
                                            debugContext=debugContext))


def div(tensors, debugContext=""):
    assert len(tensors) == 2
    if check_all_constant(tensors):
        # degrade to np op
        result = tensors[0].data / tensors[1].data
        return constant(result, debugContext=debugContext)
    return TTensor(get_builder().aiOnnx.div([t.getIpuIndex() for t in tensors],
                                            debugContext=debugContext))


def sub(tensors, debugContext=""):
    assert len(tensors) == 2
    if check_all_constant(tensors):
        # degrade to np op
        result = tensors[0].data - tensors[1].data
        return constant(result, debugContext=debugContext)
    return TTensor(get_builder().aiOnnx.sub([t.getIpuIndex() for t in tensors],
                                            debugContext=debugContext))


def max(tensor_list, debugContext=""):
    if check_all_constant(tensor_list):
        # degrade to np op
        arr_list = [t.data for t in tensor_list]
        result = np.max(arr_list)
        return constant(result)
    return TTensor(get_builder().aiOnnx.max(
        [t.getIpuIndex() for t in tensor_list], debugContext=debugContext))


def min(tensor_list, debugContext=""):
    if check_all_constant(tensor_list):
        # degrade to np op
        arr_list = [t.data for t in tensor_list]
        result = np.min(arr_list)
        return constant(result)
    return TTensor(get_builder().aiOnnx.min(
        [t.getIpuIndex() for t in tensor_list], debugContext=debugContext))


def split(x, lenOfSplit, dim, debugContext=""):
    """
    args:
        lenOfSplit: (4,1) split into two piece，one's length is 4 ,
                    the other is 1
    """
    return TTensor(get_builder().aiOnnx.split([x.getIpuIndex()],
                                              len(lenOfSplit),
                                              dim,
                                              lenOfSplit,
                                              debugContext=debugContext))


def clip(x, minmun=-np.inf, maxmun=np.inf, debugContext=""):
    if get_ai_onnx_version() >= 11:
        minmun = constant(np.asarray(minmun).astype(np.float32))
        maxmun = constant(np.asarray(maxmun).astype(np.float32))
        return TTensor(get_builder().aiOnnx.clip(
            [x.getIpuIndex(),
             minmun.getIpuIndex(),
             maxmun.getIpuIndex()],
            debugContext=debugContext))
    else:
        return TTensor(get_builder().aiOnnx.clip([x.getIpuIndex()],
                                                 maxmun.getIpuIndex(),
                                                 minmun.getIpuIndex(),
                                                 debugContext=debugContext))


def reduceprod(x, dim, keepdims=False, debugContext=""):
    """
    args:
        dim: int .which dim to do prod
    """
    x = get_builder().aiOnnx.reduceprod([x.getIpuIndex()],
                                        axes=[dim],
                                        keepdims=keepdims,
                                        debugContext=debugContext)
    return TTensor(x)


def cast(x, target_type='FLOAT', debugContext=''):
    """
    target_type:
        FLOAT|FLOAT16|INT8|INT16|INT32|UINT8|UINT16|UINT32|BOOL
    """
    target_type = 'FLOAT' if target_type == 'FLOAT32' else target_type
    if check_all_constant([x]):
        # degrade to np op
        data = x.data.astype(mappin_gc2npy[target_type])
        return constant(data)
    else:
        return TTensor(get_builder().aiOnnx.cast([x.getIpuIndex()],
                                                 target_type.upper(),
                                                 debugContext))


def log(x, debugContext=''):
    return TTensor(get_builder().aiOnnx.log([x.getIpuIndex()], debugContext))


def less(x, y, debugContext=''):
    x, y = align_tensor([x, y])
    return TTensor(get_builder().aiOnnx.less(
        [x.getIpuIndex(), y.getIpuIndex()], debugContext))


def abs(x, debugContext=''):
    return TTensor(get_builder().aiOnnx.abs([x.getIpuIndex()], debugContext))


def argmax(x, axis=0, keepdims=0, debugContext=''):

    return TTensor(get_builder().aiOnnx.argmax([x.getIpuIndex()], axis,
                                               keepdims, debugContext))


def reducemax(x, axes=0, keepdims=0, debugContext=''):
    if isinstance(axes, int):
        axes = [axes]
    assert isinstance(axes, list) or axes is None
    return TTensor(get_builder().aiOnnx.reducemax([x.getIpuIndex()], axes,
                                                  keepdims, debugContext))


def reducemin(x, axes=0, keepdims=0, debugContext=''):
    if isinstance(axes, int):
        axes = [axes]
    assert isinstance(axes, list) or axes is None
    return TTensor(get_builder().aiOnnx.reducemin([x.getIpuIndex()], axes,
                                                  keepdims, debugContext))


def reduceMean(x, axes, keepdims=False, debugContext=''):
    if isinstance(axes, int):
        axes = [axes]
    assert isinstance(axes, list) or axes is None
    return TTensor(get_builder().aiOnnx.reducemean([x.getIpuIndex()],
                                                   axes=axes,
                                                   keepdims=keepdims,
                                                   debugContext=debugContext))


def greater(x, y, debugContext=''):
    x, y = align_tensor([x, y])
    return TTensor(get_builder().aiOnnx.greater(
        [x.getIpuIndex(), y.getIpuIndex()], debugContext))


def equal(x, y, debugContext=''):
    x, y = align_tensor([x, y])
    return TTensor(get_builder().aiOnnx.equal(
        [x.getIpuIndex(), y.getIpuIndex()], debugContext))


def logical_and(x, y, debugContext=''):
    return TTensor(get_builder().aiOnnx.logical_and(
        [x.getIpuIndex(), y.getIpuIndex()], debugContext))


def logical_not(x, debugContext=''):
    return TTensor(get_builder().aiOnnx.logical_not([x.getIpuIndex()],
                                                    debugContext))


def logical_or(x, y, debugContext=''):
    return TTensor(get_builder().aiOnnx.logical_or(
        [x.getIpuIndex(), y.getIpuIndex()], debugContext))


def reduceSum(x, axes=None, keepdims=0, debugContext=''):
    if isinstance(axes, int):
        axes = [axes]
    assert isinstance(axes, list) or axes is None
    return TTensor(get_builder().aiOnnx.reducesum([x.getIpuIndex()], axes,
                                                  keepdims, debugContext))


def cumsum(x, axes, exclusive=0, reverse=0, debugContext=''):
    if x.dtype == 'FLOAT16':
        raise NotImplementedError('not support fp16')
    return TTensor(get_builder().aiOnnx.cumsum(
        [x.getIpuIndex(), axes.getIpuIndex()], exclusive, reverse,
        debugContext))


def expand(x, shape, debugContext=''):
    return TTensor(get_builder().aiOnnx.expand(
        [x.getIpuIndex(), shape.getIpuIndex()], debugContext=debugContext))


def randomuniformlike(x, high=6.0, low=-6.0):
    result = get_builder().aiOnnx.randomuniformlike([x.getIpuIndex()],
                                                    high=high,
                                                    low=low)
    return TTensor(result)


def flatten(x):
    '''implements the np.flatten function
    '''
    if check_all_constant([x]):
        x = x.data.flatten()
        return constant(x)
    x = get_builder().aiOnnx.flatten([x.getIpuIndex()], 0)
    return TTensor(x).squeeze(0)


def oneslike(t, dtype=None, debugContext=''):
    dtype = t.dtype if dtype is None else dtype
    dshape = t.pureShape
    assert 0 not in dshape
    return ones(dshape, dtype, debugContext=debugContext)


def ones(shape, dtype='FLOAT', debugContext=''):
    return constant(np.ones(shape, dtype=mappin_gc2npy[dtype]),
                    debugContext=debugContext)


def zeroslike(t, dtype=None, debugContext=''):
    dtype = t.dtype if dtype is None else dtype
    dshape = t.pureShape
    assert 0 not in dshape
    return zeros(dshape, dtype, debugContext=debugContext)


def zeros(shape, dtype='FLOAT', debugContext=''):
    return constant(np.zeros(shape, dtype=mappin_gc2npy[dtype]),
                    debugContext=debugContext)


def where(condition, x, y, debugContext=''):
    return TTensor(get_builder().aiOnnx.where(
        [condition.getIpuIndex(),
         x.getIpuIndex(),
         y.getIpuIndex()]))


def detach(x):
    return TTensor(get_builder().aiGraphcore.detach([x.getIpuIndex()]))


def one_hot(indices, depth, values=None, debugContext=''):
    '''
        values: [off_value, on_value]
        if indice is -1, the corrosponding arr is [0]*depth
    '''
    if isinstance(depth, int):
        depth = to_tensor(depth, dtype='INT64')
    if values is None:
        values = constant(np.asarray([0, 1]).astype(np.int32),
                          debugContext=debugContext)
    assert indices.dtype in [
        'INT32', 'INT64', 'INT16', 'UINT32', 'UINT64', 'UINT16'
    ]
    assert depth.dtype in [
        'INT32', 'INT64', 'INT16', 'UINT32', 'UINT64', 'UINT16'
    ]
    assert values.dtype in [
        'INT32', 'INT64', 'INT16', 'UINT32', 'UINT64', 'UINT16'
    ]
    result = get_builder().aiOnnx.onehot(
        [indices.getIpuIndex(),
         depth.getIpuIndex(),
         values.getIpuIndex()],
        debugContext=debugContext)
    result = TTensor(result)
    result_shape = list(result.pureShape)
    if result_shape[1] == 0:
        result_shape[1] = depth
    result = result.reshape(result_shape)
    return result


def tile(input, repeats, debugContext=""):
    if check_all_constant([input, repeats]):
        result = np.tile(input.data, repeats.data)
        return constant(result)
    result = get_builder().aiOnnx.tile(
        [input.getIpuIndex(), repeats.getIpuIndex()], debugContext)
    return TTensor(result)


def checkTensorsTypeSame(tensors_list):
    types = [t.dtype for t in tensors_list]
    if len(set(types)) == 2:
        assert 'int32' in types and 'int64' in types, 'only int32 and int64 seem as same type'
    else:
        assert len(set(types)) == 1, 'types should be same'


def np64to32_or_16(np_arr):
    # temp ussage while IPU not support with 64bit data
    local_mappin = {
        'int64': np.int32,
        'uint64': np.uint32,
        'float64': np.float16
    }
    if np_arr.dtype.name in list(local_mappin.keys()):
        np_arr = np_arr.astype(local_mappin[np_arr.dtype.name])
    return np_arr


def to_tensor(x, dtype=None):
    # return ConstantTensor if x type is int,float,ndarray,list, or ConstantTensor
    # return TTensor if x type is TTensor
    if np.isscalar(x) or isinstance(x, list):
        # if type(x) in [int,float,list]:
        x = np.array(
            x
        )  # python basic type int and float should be converted to np.float32 and np.int32
        x = np64to32_or_16(x)
        if dtype is not None:
            x = x.astype(mappin_gc2npy[dtype])
        x = constant(x)
    elif type(x) in [np.ndarray]:
        if dtype is not None:
            x = x.astype(mappin_gc2npy[dtype])
        x = constant(x)
    elif isinstance(x, TTensor):
        pass
    else:
        raise TypeError('not a legal type')
    return x


TYPE_SEQUENCE = {'BOOL': 0, "UINT": 1, "INT": 2, "FLOAT": 3}


def align_tensor(tensors):
    tensors = [to_tensor(t) for t in tensors]
    _types = [t.dtype.upper() for t in tensors]
    bits_list = [mappin_type2bits[t.dtype] for t in tensors]
    nakedTypes = [t.dtype.strip(string.digits).upper() for t in tensors]
    type_idx = 0
    dst_dtype = 'BOOL'
    for dtype in nakedTypes:
        if TYPE_SEQUENCE[dtype] > type_idx:
            type_idx = TYPE_SEQUENCE[dtype]
            dst_dtype = dtype
    max_bits = builtins.max(bits_list)
    if 'FLOAT32' in _types or 'FLOAT' in _types:
        max_bits = 32
    elif 'FLOAT16' in _types:
        max_bits = 16
    dst_dtype = dst_dtype + str(max_bits)
    new_tensors = []
    for t in tensors:
        t = t.cast(dst_dtype)
        new_tensors.append(t)
    tensors = new_tensors
    if TTensor in [type(t) for t in tensors]:
        tensors = [TTensor(t.getIpuIndex()) for t in tensors]
    return tensors


def int32toint64(tensor):
    return t.cast('INT64') if t.type == 'int32' else t


def sum_list(l):
    current = 1
    for ele in l:
        current *= ele
    return current


class TTensor():
    def __init__(self, name, nodata=False):
        assert isinstance(name, str)
        TENSOR_NAMES.append(name)
        self.__name = name
        if safe_mode() and not nodata:
            assert isinstance(self.pureShape, (list, tuple))
            assert isinstance(self.dtype, str)

    def copy_from_tensor(self, tensor):
        assert self.__class__.__name__ == tensor.__class__.__name__
        self.__name = tensor.__name

    def getIpuIndex(self, ):
        return self.__name

    @property
    def name(self, ):
        return self.getIpuIndex()

    def __len__(self, ):
        return self.pureShape[0]

    def numel(self, ):
        return sum_list(self.pureShape)

    def unsqueeze(
        self,
        dims,
    ):
        """
        args:
            dim: list of int of which dim will delete
                    eg:[3] or [1,3]
        """
        if isinstance(dims, int):
            dims = [dims]
        result = unsqueeze(self, dims=dims)
        return result

    def squeeze(
        self,
        dims,
    ):
        """
        args:
            dim: list of int of which dim will delete
                    eg:[3] or [1,3]
        """
        if isinstance(dims, int):
            dims = [dims]
        result = squeeze(self, dims=dims)
        return result

    def cast(self, type_str):
        result = cast(self, type_str.upper())
        return result

    def add(self, tensors):
        checkTensorsTypeSame(tensors)
        result = add(tensors)
        return result

    def __add__(self, other):
        tensors = align_tensor([self, other])
        return self.add(tensors)

    def __radd__(self, other):
        tensors = align_tensor([other, self])
        return self.add(tensors)

    def sub(self, tensors):
        checkTensorsTypeSame(tensors)
        result = sub(tensors)
        return result

    def __sub__(self, other):
        tensors = align_tensor([self, other])
        return self.sub(tensors)

    def __neg__(self):
        return self.__rsub__(0)

    def __rsub__(self, other):
        tensors = align_tensor([other, self])
        return self.sub(tensors)

    def mul(self, tensors):
        checkTensorsTypeSame(tensors)
        result = mul(tensors)
        return result

    def __mul__(self, other):
        tensors = align_tensor([self, other])
        return self.mul(tensors)

    def __rmul__(self, other):
        tensors = align_tensor([other, self])
        return self.mul(tensors)

    def truediv(self, tensors):
        checkTensorsTypeSame(tensors)
        result = div(tensors)
        return result

    def __truediv__(self, other):
        tensors = align_tensor([self, other])
        return self.truediv(tensors)

    def __rtruediv__(self, other):
        tensors = align_tensor([other, self])
        return self.truediv(tensors)

    def __repr__(self, ):
        string = self.__class__.__name__ + ': ' + self.__name + ', shape: ' + str(
            self.pureShape) + ', dtype: ' + self.dtype
        string += ', ID: ' + str(id(self))
        return string

    def __format__(self, code):
        return self.__repr__()

    @property
    def pureShape(self):
        return get_builder().getTensorShape(self.__name)

    @property
    def shape(self):
        self_shape = self.pureShape
        self_shape_np = np.array(self_shape, np.uint32)
        input_shape_tensor = shapeConstant(self_shape_np)
        return input_shape_tensor

    @property
    def dtype(self):
        type_str = get_builder().getTensorDtypeString(self.__name).upper()
        if type_str == 'FLOAT32':
            return 'FLOAT'
        return type_str

    def get_shape(self):
        return self.shape

    def detach(self, ):
        return detach(self)

    def flatten(self, ):
        return flatten(self)

    def transpose(self, dim_order):
        return transpose(self, dim_order)

    def reshape(self, target_shape):
        return reshape(self, target_shape)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )

        def get_start(start):
            return 0 if start is None else start

        def get_stop(stop, dim):
            return self.pureShape[dim] if stop is None else stop

        axes = []
        starts = []
        ends = []
        squeezes = []
        for dim, slice_obj in enumerate(index):
            if getattr(slice_obj, 'step', None) is not None:
                raise NotImplementedError(
                    "Tensor indexing don't support interval currently")
            if isinstance(slice_obj, int):
                start = slice_obj
                end = slice_obj + 1
                squeezes.append(dim)
            elif isinstance(slice_obj, ConstantTensor):
                start = int(slice_obj.data)
                end = start + 1
            else:
                start = get_start(slice_obj.start)
                start = int(getattr(start, 'data', start))
                end = get_stop(slice_obj.stop, dim)
                end = int(getattr(end, 'data', end))
            axes.append(dim)
            starts.append(start)
            ends.append(end)

        axes = constant(np.array(axes))
        starts = constant(np.array(starts))
        ends = constant(np.array(ends))

        # do slice in onnx
        output = gc_slice(self, axes, starts, ends)
        if len(squeezes) > 0:
            output = squeeze(output, dims=squeezes)

        return output


class ConstantTensor(TTensor):
    def __init__(self, name, data):
        assert isinstance(data, np.ndarray)
        if data.dtype.name in ['float64', 'float']:
            data = data.astype(np.float32)
        self.data = data
        if name is None:
            self.__name = 'uninitialized_constant:' + str(CONSTANT_COUNTER[0])
            CONSTANT_COUNTER[0] = CONSTANT_COUNTER[0] + 1
            self.initialized = False
        else:
            super().__init__(name)
            self.__name = name

    def copy_from_tensor(self, tensor):
        assert self.__class__.__name__ == tensor.__class__.__name__
        self.__name = tensor.__name
        self.data = tensor.data
        self.initialized = tensor.initialized

    def real_init(self, ):
        assert not self.initialized
        self.data = np.ascontiguousarray(self.data.copy())
        name = get_builder().aiOnnx.constant(self.data)
        super().__init__(name)
        self.__name = name  # private attribute can not be inherited
        self.initialized = True

    def getIpuIndex(self, ):
        if self.initialized is False:
            self.real_init()
        name = super().getIpuIndex()
        assert name is not None
        return name

    @property
    def pureShape(self):
        return self.data.shape

    @property
    def dtype(self):
        result = self.data.dtype.name.upper()
        if result == 'FLOAT32':
            result = 'FLOAT'
        return result

    def as_list(self, ):
        return self.data.tolist()

    def __repr__(self, ):
        string = self.__class__.__name__ + ': ' + self.__name + ', shape: ' + str(
            self.pureShape) + ', dtype: ' + self.dtype
        string = string + ', constant: ' + str(self.data)
        string += ', ID: ' + str(id(self))
        return string

    def __getitem__(self, index):
        if isinstance(index, int):
            return constant(self.data[index])

        if not isinstance(index, tuple):
            index_tuple = (index, )
        else:
            index_tuple = index

        index_list = []
        for index in index_tuple:
            if isinstance(index, int):
                index = index
                index_list.append(index)
                continue
            else:
                index = [index.start, index.stop, index.step]
            slices = []
            for ele in index:
                if isinstance(ele, list):
                    ele = [scalarTensor2int(ele_ele) for ele_ele in ele]
                elif ele is None:
                    pass
                else:
                    ele = scalarTensor2int(ele)
                slices.append(ele)
            index_list.append(slice(*slices))

        return constant(self.data[tuple(index_list)])


class ShapeTensor(ConstantTensor):
    def __init__(self, name, data):
        assert len(data.shape) == 1  # ShapeTensor rank is 1
        super().__init__(name, data)
        self.__name = name

    @property
    def ndims(self):
        return self.numel()


def eliminate_constantTensor_from_list(num_list, to_type):
    results = []
    for num in num_list:
        if isinstance(num, ConstantTensor):
            num = to_type(num.data)
        results.append(num)
    return results


def scalar2num(tensor):
    return t.data if isinstance(t, ConstantScalar) else t


class ConstantScalar(ConstantTensor):
    def __init__(self, name, data):
        assert np.isscalar(data) or data.size == 1 or isinstance(
            data, (int, float))
        data = np.array(np.asscalar(np.asarray(data).flatten()))
        super().__init__(name, data)
        self.__name = name

    def __eq__(self, other):
        return scalar2num(self) == scalar2num(other)

    def __ge__(self, other):
        return scalar2num(self) >= scalar2num(other)

    def __le__(self, other):
        return scalar2num(self) <= scalar2num(other)

    def __lt__(self, other):
        return scalar2num(self) < scalar2num(other)

    def __gt__(self, other):
        return scalar2num(self) > scalar2num(other)

    def __ne__(self, other):
        return scalar2num(self) > scalar2num(other)


def nllloss(prob,
            label,
            reductionType=popart.ReductionType.Mean,
            debugPrefix=''):
    #
    with name_scope(debugPrefix):
        loss = get_builder().aiGraphcore.nllloss(
            [prob.getIpuIndex(), label.getIpuIndex()],
            reductionType,
            debugContext=debugPrefix)
    return TTensor(loss)


def get_all_params():
    all_params = [
        TTensor(name) for name in get_builder().getInputTensorIds()
        if get_builder().isInitializer(name)
    ]
    return all_params


mappin_type2bits = {
    'FLOAT': 32,
    'FLOAT16': 16,
    "UINT8": 8,
    "UINT16": 16,
    "UINT32": 32,
    "UINT64": 64,
    "INT8": 8,
    "INT16": 16,
    "INT32": 32,
    "INT64": 64,
    "BOOL": 8
}

mappin_gc2npy = {
    'float': np.float32,
    'float32': np.float32,
    'float16': np.float16,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'bool': np.bool,
    'FLOAT': np.float32,
    'FLOAT32': np.float32,
    'FLOAT16': np.float16,
    'INT8': np.int8,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
    'UINT16': np.uint16,
    'UINT32': np.uint32,
    'BOOL': np.bool,
}


# optimizer
class SGD:
    def __init__(self,
                 learning_rate=0.001,
                 momentum=0.0005,
                 weight_decay=0.0001,
                 use_locking=False,
                 name='Momentum',
                 use_nesterov=False,
                 clip_norm=None,
                 lossScaling=1.0,
                 specific_dic={}):
        assert use_locking is False
        assert use_nesterov is False
        self.learning_rate = learning_rate
        self.name = name
        self.clip_norm = clip_norm
        self.lossScaling = lossScaling
        self.opti_cfg = {
            "defaultLearningRate": (self.learning_rate, False),
            "defaultMomentum": (momentum, True),
            "defaultWeightDecay": (weight_decay, True),
        }
        if self.lossScaling != 1.0:
            self.opti_cfg['lossScaling'] = (self.lossScaling, True)
        if clip_norm is not None:
            print('clip norm gradients:', clip_norm)
            self.gc_optimizer = popart.SGD(
                self.opti_cfg,
                clip_norm_settings=[
                    popart.ClipNormSettings.clipAllWeights(clip_norm)
                ])
        else:
            self.gc_optimizer = popart.SGD(self.opti_cfg)
        for name in specific_dic:
            self.gc_optimizer.insertSpecific(name, specific_dic[name])

    def adj_lr(self, lr, sess, specific_dic={}):
        self.opti_cfg['defaultLearningRate'] = (lr, False)
        new_optimizer = popart.SGD(self.opti_cfg)
        for name in specific_dic:
            new_optimizer.insertSpecific(name, specific_dic[name])
        sess.updateOptimizerFromHost(new_optimizer)
        self.gc_optimizer = new_optimizer


def deduce_half(input, fp16_on):
    # Derive whether to use half-precision according to the input and the given parameters: fp16_on
    assert fp16_on in [False, True, None]
    # False: fp32
    # True: fp16
    # None: depend on input
    # return :
    #   cast flag: False(no cast), fp16/fp32(cast output to fp16/fp32)
    #   input: casted input
    #   fp16_on: True(cast weights to fp16), False(cast weights to fp32)
    cast_flag = False
    if input.dtype.lower() == 'float16':
        if fp16_on is None:
            fp16_on = True
        elif fp16_on is False:
            input = input.cast('FLOAT32')
            cast_flag = 'FLOAT16'
            # pass
    elif input.dtype.lower() in ['float32', 'float']:
        if fp16_on:
            input = input.cast('FLOAT16')
            cast_flag = 'FLOAT32'
            # pass
    else:
        raise RuntimeError('input wrong type: {}'.format(input.dtype))
    return cast_flag, input, fp16_on
