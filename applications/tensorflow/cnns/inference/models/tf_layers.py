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

from typing import Any, List, Optional, Tuple, Union

import tensorflow.compat.v1 as tf
from tensorflow import contrib
from tensorflow.python.ipu.ops import normalization_ops
from tensorflow_core.python.keras.backend import unique_object_name


def conv(input_tensor: tf.Tensor, kernel_size: Union[int, Tuple[int, int]], filters_out: int, stride: Optional[int] = 1,
         padding: Optional[str] = 'SAME', add_bias: Optional[bool] = True, dtype: Optional[Any] = tf.float16,
         name: Optional[str] = None, weight_suffix: Optional[str] = "kernel", bias_suffix: Optional[str] = "conv/bias",
         *_):
    """Apply conv and optional bias on input tensor with TensorFlow.

    Args:
        input_tensor: Input data
        kernel_size: Filter size (assumes equal height and width)
        filters_out: Number of output filters
        stride: Stride of the filter
        padding: Type of padding to use
        add_bias: Should bias be added
        dtype: Data type of parameters
        name: Optional name for this op

    Returns: Output of convolution operator.
    """

    # Assumes input in NHWC format.
    filters_in = input_tensor.get_shape()[-1]
    if isinstance(kernel_size, int):
        w_shape = [kernel_size, kernel_size, filters_in, filters_out]
    else:
        w_shape = kernel_size + (filters_in, filters_out)
    w_init = contrib.layers.xavier_initializer(dtype=dtype)
    if name is None:
        name = unique_object_name("conv2d", zero_based=True)

    name_scope = tf.get_default_graph().get_name_scope()
    if name_scope not in ["", None]:
        name = name_scope + "/" + name

    with tf.get_default_graph().as_default():
        with tf.variable_scope(name):
            weights = tf.get_variable(weight_suffix, shape=w_shape, initializer=w_init, dtype=dtype)

    output_tensor = tf.nn.conv2d(input_tensor, weights, [1, stride, stride, 1], padding=padding.upper(),
                                 name=name)

    if add_bias:
        b_shape = [filters_out]
        b_init = tf.zeros_initializer()
        with tf.variable_scope(name):
            biases = tf.get_variable(bias_suffix, shape=b_shape, initializer=b_init, dtype=dtype)
        output_tensor += biases
    return output_tensor


def relu(input_tensor: tf.Tensor, name: Optional[str] = 'relu', max_value: Optional[float] = None, *_):
    """Apply rectified linear operation on input.

    Args:
      input_tensor: Input data
      name: Optional name prefix for this op
      max_value: Optional max_value to clip output at.

    Returns: `Tensor` of same shape as `input_tensor`, after applying ReLu
    """
    if max_value is not None:
        return tf.minimum(max_value, tf.nn.relu(input_tensor, name=f'{name}_relu'))
    return tf.nn.relu(input_tensor, name=f'{name}_relu')


def reshape(input_tensor: tf.Tensor, shape: Tuple, name: Optional[str] = 'reshape', *_):
    """Reshape tensor

    Args:
        input_tensor: Input data
        shape: size of output tensor
        name: Optional name prefix for this op
    """
    return tf.reshape(tensor=input_tensor, shape=shape, name=name)


def norm(input_tensor: tf.Tensor, norm_type: Optional[str] = 'BATCH',
         groups: Optional[int] = 32, training: Optional[bool] = False,
         name: Optional[str] = None, epsilon: Optional[float] = 1.001e-5,
         momentum: Optional[float] = 0.997, axis: Optional[int] = -1,
         center: Optional[bool] = True,
         scale: Optional[bool] = True, *_):
    """Apply norm ops.

    Args:
        input_tensor: Input data
        norm_type: Type of normalization to be applied - ('BATCH' or 'GROUP')
        groups: Number of channel groups over which stats are computed
        training: True if network is constructed for training
        name: Optional name prefix for this op
        epsilon: Small float added to variance to avoid dividing by zero.
        momentum: Momentum for the moving average.
        axis: An `int`, the axis that should be normalized (typically the features
        axis).
        center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
        scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.

    Returns:
        A `Tensor` representing the output of the operation, same shape as input_tensor

    Raises:
        ValueError if unsupported norm_type is requested
    """

    if norm_type == 'BATCH':
        if not name:
            name = unique_object_name("bn", zero_based=True)
        else:
            name = unique_object_name(name, zero_based=True)
        name_scope = tf.get_default_graph().get_name_scope()
        if name_scope not in (None, ""):
            name = name_scope + "/" + name
        output_tensor = tf.keras.layers.BatchNormalization(axis=axis, fused=True,
                                                           center=center, scale=scale,
                                                           trainable=training, momentum=momentum,
                                                           epsilon=epsilon, name=name)(input_tensor, training=training)
    elif norm_type == 'GROUP':
        if not name:
            name = unique_object_name("gn", zero_based=True)
        output_tensor = normalization_ops.group_norm(input_tensor, groups=groups,
                                                     center=center, scale=scale,
                                                     training=training, trainable=training,
                                                     channels_axis=-1, scope=name)
    else:
        raise ValueError('norm_type must be "BATCH" or "GROUP", got %s instead' % norm_type)
    return output_tensor


def concat(input_tensors: List[tf.Tensor], axis: Optional[int] = -1, name: Optional[str] = '', *_):
    """Concatenates tensors along one dimension.

    Args:
        input_tensors: A list of `Tensor` objects
        axis: Optional dimension along which to concatenate. Must be
          in the range `[-rank(values), rank(values))`, defaults to axis=-1 (channel dim)
        name: Optional name prefix for this operation

    Returns:
        A `Tensor` resulting from concatenation of the input tensors.
    """
    if name == '':
        name = unique_object_name("concat")
    else:
        name = f'{name}_concat'
    return tf.concat(input_tensors, tf.constant(axis, name=name + '_axis'), name=name)


def avg_pool(input_tensor: tf.Tensor, kernel_size: int, strides: int, name: Optional[str] = 'avg_pool',
             padding: Optional[str] = 'VALID',
             *_) -> tf.Tensor:
    """Performs average pooling on the input.

    Each entry in `output` is the mean of the corresponding kernel_size x kernel_size
    window in `input_tensor`.

    Args:
        input_tensor:  A 4-D `Tensor` of shape `[batch, height, width, channels]`
        kernel_size:  The size of the window along height and width of the input tensor
        strides: The stride of the sliding window for width and height
        name: Optional name for the operation
        padding: 'SAME' or 'VALID'

    Returns:
        A `Tensor` with the same type as `input_tensor`.  The average pooled output tensor,
        with same batch and channel dimension as `input_tensor`.
    """
    return tf.nn.avg_pool(input_tensor, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, strides, strides, 1], padding=padding.upper(), name=name)


def global_avg_pool(input_tensor: tf.Tensor, name: Optional[str] = 'global_avg_pool', *_) -> tf.Tensor:
    """Performs global average pooling on the input

    Args:
        input_tensor: a 4-D 'Tensor' of shape `[batch, height, width, channels]'
        name: Optional name of the operation

    Returns:
        A 'Tensor' with the same type as `input_tensor` with shape (batch_size, channels).

    """
    return tf.reduce_mean(input_tensor, axis=[1, 2], name=name)


def max_pool(input_tensor: tf.Tensor, kernel_size: int, strides: Optional[int] = None, name: Optional[str] = 'max_pool',
             padding: Optional[str] = 'VALID',
             *_) -> tf.Tensor:
    """Performs max pooling on the input.

    Each entry in `output` is the max of the corresponding kernel_size x kernel_size
    window in `input_tensor`.

    Args:
        input_tensor:  A 4-D `Tensor` of shape `[batch, height, width, channels]`
        kernel_size:  The size of the window along height and width of the input tensor
        strides: The stride of the sliding window along width and height
        name: Optional name for the operation
        padding: 'SAME' or 'VALID'

    Returns:
        A `Tensor` with the same type as `input_tensor`.  The max pooled output tensor,
        with same batch and channel dimension as `input_tensor`.
    """
    if strides is None:
        strides = kernel_size - 1
    return tf.nn.max_pool(input_tensor, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, strides, strides, 1], padding=padding.upper(), name=name)


def fully_connected(input_tensor: tf.Tensor, num_outputs: int, dtype: Optional[Any] = tf.float16,
                    name: Optional[str] = 'fc', *_) -> tf.Tensor:
    """Applies fully connected layer to `input_tensor`.

    Args:
        input_tensor: 2D Tensor of dimensions [batch, in_units]
        num_outputs: Number of output units
        dtype: Data type of parameters
        name: Optional name for this operation

    Returns:
        A 2-D Tensor computing matmul(x, weights) + biases, dimensions [batch, num_outputs]
    """
    num_inputs = input_tensor.get_shape()[1]
    w_init = contrib.layers.xavier_initializer(dtype=dtype)
    b_init = tf.constant_initializer(0.0, dtype=dtype)

    with tf.variable_scope(name):
        weights = tf.get_variable('kernel', shape=[num_inputs, num_outputs], initializer=w_init, dtype=dtype)
        biases = tf.get_variable('bias', shape=[num_outputs], initializer=b_init, dtype=dtype)

    return tf.nn.xw_plus_b(input_tensor, weights, biases, name=name)


def zero_padding(input_tensor: tf.Tensor, padding: Tuple[Tuple[int, int], Tuple[int, int]],
                 name: Optional[str] = 'zero_padding2d', *_) -> tf.Tensor:
    """Zero-padding layer for images.

    Adds rows and columns of zeros at the top, bottom, left and right side of an image tensor.

    Args:
        input_tensor: 4D Tensor with shape `(batch, rows, cols, channels)`
        padding: Padding - `((top_pad, bottom_pad), (left_pad, right_pad))`
        name: Optional name for the op.

    Returns:
        Padded tensor of dims `(batch, padded_rows, padded_cols, channels)`

    """
    paddings = tf.constant(((0, 0),) + padding + ((0, 0),))
    return tf.pad(input_tensor, paddings, name=name)


def softmax(input_tensor: tf.Tensor, axis: Optional[int] = -1, name: Optional[str] = 'softmax', *_) -> tf.Tensor:
    """Computes softmax activations.

    Args:
        input_tensor: Input logits.
        axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name: Optional name for the operation.

    Returns:
        `Tensor` that has the same type and shape as `input_tensor`.

    """
    return tf.nn.softmax(input_tensor, axis=axis, name=name)


def squeeze(input_tensor: tf.Tensor, axis: Optional[List], name: Optional[str] = 'squeeze', *_) -> tf.Tensor:
    """Squeeze dims of size 1 from input_tensor.

    Args:
        input_tensor:  A `Tensor`. The `input` to squeeze
        axis: An optional list of `ints`. Defaults to `[]`.
              If specified, only squeezes the dimensions listed. The dimension
              index starts at 0. It is an error to squeeze a dimension that is not 1.
              Must be in the range `[-rank(input), rank(input))`
        name: Optional name for the op.

    Returns:
            A `Tensor`. Has the same type as `input`. Contains the same data as `input`, but has one or more
            dimensions of size 1 removed.

    """
    return tf.squeeze(input_tensor, axis=axis, name=name)


def conv_norm_relu(input_tensor: tf.Tensor,
                   filters: int,
                   kernel_height: int,
                   kernel_width: Optional[int] = None,
                   padding: Optional[str] = 'SAME',
                   strides: Optional[int] = 1,
                   name: Optional = None,
                   norm_type: Optional[str] = 'BATCH',
                   norm_suffix: Optional[str] = '_bn',
                   conv_suffix: Optional[str] = '_conv',
                   weight_suffix: Optional[str] = 'kernel'):
    """Utility function to apply conv + norm + relu.

    Args:
        input_tensor: input tensor.
        filters: filters in `Conv2D`.
        kernel_height: Height of the convolution kernel.
        kernel_width: Width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution,  `name + '_bn'` for the
            batch norm layer and `name + '_relu'` for relu layer.
        norm_type: either `BATCH` for batch norm or `GROUP` for group norm.

    Returns:
        Output tensor after applying `Conv2D`, `Normalization` and `Relu`.
    """
    if kernel_width is None:
        kernel_width = kernel_height

    if name is not None:
        if norm_type == 'BATCH':
            norm_name = name + norm_suffix
        else:
            norm_name = name + '_gn'
        conv_name = name + conv_suffix
        relu_name = name + '_relu'
    else:
        norm_name = "batch_normalization" if norm_type == 'BATCH' else "group_normalization"
        conv_name = unique_object_name("conv2d", zero_based=True)
        relu_name = unique_object_name("relu", zero_based=True)

    x = conv(input_tensor,
             (kernel_height, kernel_width), filters,
             strides,
             padding,
             add_bias=False,
             name=conv_name,
             weight_suffix=weight_suffix)
    x = norm(x, axis=-1, scale=False, name=norm_name, norm_type=norm_type)
    x = relu(x, name=relu_name)
    return x


def separable_conv(input_tensor: tf.Tensor, kernel_size: Union[int, Tuple[int, int]], filters_out: int,
                   stride: Optional[int] = 1,
                   padding: Optional[str] = 'SAME', add_bias: Optional[bool] = True, dtype: Optional[Any] = tf.float16,
                   name: Optional[str] = None, *_):
    """Apply separable conv and optional bias on input tensor with TensorFlow.

      Performs a depthwise convolution that acts separately on channels followed by
      a pointwise convolution that mixes channels.  Note that this is separability
      between dimensions `[1, 2]` and `3`, not spatial separability between
      dimensions `1` and `2`.

    Args:
        input_tensor: Input data
        kernel_size: Filter size (assumes equal height and width)
        filters_out: Number of output filters
        stride: Stride of the filter
        padding: Type of padding to use
        add_bias: Should bias be added
        dtype: Data type of parameters
        name: Optional name for this op

    Returns: Output of convolution operator.
    """

    # Assumes input in NHWC format.
    filters_in = input_tensor.get_shape()[-1]
    if isinstance(kernel_size, int):
        depthwise_kernel_shape = [kernel_size, kernel_size, filters_in, 1]
    else:
        depthwise_kernel_shape = kernel_size + (filters_in, 1)
    w_init = contrib.layers.xavier_initializer(dtype=dtype)

    pointwise_kernel_shape = [1, 1, filters_in, filters_out]

    name_scope = tf.get_default_graph().get_name_scope()
    if name_scope not in ["", None]:
        name = name_scope + "/" + name

    with tf.get_default_graph().as_default():
        with tf.variable_scope(name):
            depthwise_kernel = tf.get_variable('depthwise_kernel', shape=depthwise_kernel_shape, initializer=w_init,
                                               dtype=dtype)
            pointwise_kernel = tf.get_variable('pointwise_kernel', shape=pointwise_kernel_shape, initializer=w_init,
                                               dtype=dtype)

    output_tensor = tf.nn.separable_conv2d(input_tensor, depthwise_kernel, pointwise_kernel,
                                           strides=[1, stride, stride, 1],
                                           padding=padding.upper())

    if add_bias:
        b_shape = [filters_out]
        b_init = tf.zeros_initializer()
        with tf.variable_scope(name):
            biases = tf.get_variable('conv/bias', shape=b_shape, initializer=b_init, dtype=dtype)
        output_tensor += biases
    return output_tensor


def depthwise_conv(input_tensor: tf.Tensor, kernel_size: Union[int, Tuple[int, int]],
                   filters_out: Optional[int] = None, stride: Optional[int] = 1,
                   padding: Optional[str] = 'SAME', add_bias: Optional[bool] = True, dtype: Optional[Any] = tf.float16,
                   name: Optional[str] = None, *_):
    """Apply depthwise conv and optional bias on input tensor with TensorFlow.

      Performs a depthwise convolution

    Args:
        input_tensor: Input data
        kernel_size: Filter size (assumes equal height and width)
        filters_out: Number of output filters
        stride: Stride of the filter
        padding: Type of padding to use
        add_bias: Should bias be added
        dtype: Data type of parameters
        name: Optional name for this op

    Returns: Output of convolution operator.
    """

    # Assumes input in NHWC format.
    filters_in = input_tensor.get_shape()[-1]
    if isinstance(kernel_size, int):
        depthwise_kernel_shape = [kernel_size, kernel_size, filters_in, 1]
    else:
        depthwise_kernel_shape = kernel_size + (filters_in, 1)
    w_init = contrib.layers.xavier_initializer(dtype=dtype)

    name_scope = tf.get_default_graph().get_name_scope()
    if name_scope not in ["", None]:
        name = name_scope + "/" + name

    with tf.get_default_graph().as_default():
        with tf.variable_scope(name):
            depthwise_kernel = tf.get_variable('depthwise_kernel', shape=depthwise_kernel_shape, initializer=w_init,
                                               dtype=dtype)

    output_tensor = tf.nn.depthwise_conv2d(input_tensor, depthwise_kernel,
                                           strides=[1, stride, stride, 1],
                                           padding=padding.upper())

    if add_bias:
        if filters_out:
            b_shape = [filters_out]
        else:
            b_shape = [filters_in]
        b_init = tf.zeros_initializer()
        with tf.variable_scope(name):
            biases = tf.get_variable('conv/bias', shape=b_shape, initializer=b_init, dtype=dtype)
        output_tensor += biases
    return output_tensor


def crop(input_tensor: tf.Tensor, cropping: Tuple[Tuple[int, int], Tuple[int, int]]):
    """Crop input along width and height dimensions, assumes channels_last.

    Args:
        input_tensor: Input to be cropped.
        cropping: Start and stop index along height and width.

    Returns: Cropped tensor.
    """
    _, rows, cols, _ = input_tensor.get_shape().as_list()
    return input_tensor[:, cropping[0][0]:rows - cropping[0][1], cropping[1][0]:cols - cropping[1][1], :]
