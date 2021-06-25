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

import tensorflow as tf
from collections import namedtuple
from functools import partial
from tensorflow.python.ipu import normalization_ops
from .batch_norm import batch_norm
from .model_base import ModelBase


class ResNetBase(ModelBase):
    """Base model for ResNet and ResNeXt"""

    def __init__(self, opts, definition, is_training=True):
        super().__init__(opts, is_training)

        # Apply options to layers
        self.conv = partial(conv, seed=opts["seed"])

        self.norm = partial(norm, opts=opts, is_training=is_training)
        self.norm_2 = partial(norm, opts=opts, is_training=is_training, zero_gamma_init = True)
        self.norm_dict = {'one_init': self.norm, 'zero_init': self.norm_2}
        self.fc = partial(fc, seed=opts["seed"])

        # Apply changed layers to block functions
        self.initial_block_fn = partial(
            initial_block, conv=self.conv, norm=self.norm
        )

        self.initial_block_filters = definition.initial_block_filters
        self.block_counts = definition.block_counts
        self.out_filters = definition.out_filters

        self.out_filters = [
            int(f) if not isinstance(f, tuple) else tuple(int(f2) for f2 in f)
            for f in self.out_filters
        ]

        self.block_fn = partial(definition.block_fn,
                                shortcut_type=opts["shortcut_type"],
                                conv=self.conv,
                                norm=self.norm_dict)

        # Apply dataset specific changes
        if opts["dataset"] == "imagenet":
            self.initial_block_fn = partial(self.initial_block_fn, ksize=7, initial_downsample=True)
        elif opts["dataset"] == "cifar-10":
            self.initial_block_fn = partial(self.initial_block_fn, ksize=3, initial_downsample=False)
        elif opts["dataset"] == "cifar-100":
            self.initial_block_fn = partial(self.initial_block_fn, ksize=3, initial_downsample=False)

    def _build_function_list(self):
        fn_list = []
        fn_list.append(
            partial(
                self.initial_block_fn,
                filters=self.initial_block_filters,
                name="b0"
            )
        )

        for n in range(len(self.block_counts)):
            first_stride = 1 if n is 0 else 2
            for i in range(self.block_counts[n]):
                stride = first_stride if (i == 0) else 1
                fn_list.append(
                    partial(
                        self.block_fn,
                        stride=stride,
                        filters=self.out_filters[n],
                        name="b{}/{}".format(n + 1, i),
                        use_shortcut=(i == 0)
                    )
                )
                fn_list.append(
                    partial(
                        final_block_relu, name="b{}/{}/relu".format(n + 1, i)
                    )
                )
        fn_list.append(
            partial(
                tf.reduce_mean, reduction_indices=[1, 2], name="reduce_mean"
            )
        )
        fn_list.append(
            partial(self.fc, num_units_out=self.num_classes, name="fc1")
        )
        return fn_list


def norm(x, opts, is_training=True, zero_gamma_init = False):
    # We initialise the parameters depending on the gamma_init argument
    if zero_gamma_init:
        p_init = {'gamma': tf.zeros_initializer(), 'beta': tf.zeros_initializer()}
    else:
        p_init = {'gamma': tf.ones_initializer(), 'beta': tf.zeros_initializer()}

    norm_type = (
        "GROUP"
        if opts["group_norm"]
        else "BATCH"
        if opts["batch_norm"]
        else None
    )
    if norm_type == "BATCH":
        x = batch_norm(
            x,
            center=True,
            scale=True,
            training=is_training,
            trainable=True,
            epsilon=1e-5,
            gamma_initializer = p_init['gamma'],
            beta_initializer = p_init['beta']
        )
    elif norm_type == "GROUP":
        # Adding arguments to group norm
        x = normalization_ops.group_norm(
            x,
            groups=opts["groups"],
            param_initializers = p_init
        )

    tf.add_to_collection("activations", x)
    return x


def fc(x, num_units_out, name, seed=None):
    with tf.variable_scope(name, use_resource=True):
        x = tf.layers.dense(
            inputs=x,
            units=num_units_out,
            kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
        )
        tf.add_to_collection("activations", x)
        return x


def max_pool(x, ksize=3, stride=2):
    x = tf.nn.max_pool(
        x,
        ksize=[1, ksize, ksize, 1],
        strides=[1, stride, stride, 1],
        padding="SAME",
    )
    tf.add_to_collection("activations", x)
    return x


def conv(x, ksize, stride, filters_out, groups=1, seed=None):
    in_filters = x.get_shape().as_list()[3]
    with tf.variable_scope("conv", use_resource=True):
        W = tf.get_variable(
            "conv2d/kernel",
            shape=[ksize, ksize, in_filters / groups, filters_out],
            dtype=x.dtype,
            trainable=True,
            initializer=tf.variance_scaling_initializer(seed=seed),
        )
        return tf.nn.conv2d(
            x,
            filters=W,
            strides=[1, stride, stride, 1],
            padding="SAME",
            data_format="NHWC",
        )


def initial_block(x, filters, conv, norm, ksize=3, initial_downsample=False, name="b0"):
    with tf.variable_scope(name, use_resource=True):
        x = conv(x, ksize, 2 if initial_downsample else 1, filters)
        x = norm(x)
        x = tf.nn.relu(x)
        if initial_downsample:
            x = max_pool(x, ksize=3, stride=2)
        return x


def final_block_relu(x, name):
    with tf.variable_scope(name, use_resource=True):
        return tf.nn.relu(x)


# Shortcut types:
# A: Zero padding when increasing dims
# B: 1x1 conv when increasing dims
# C: 1x1 conv on all shortcuts
def block2(x, stride, filters, name, use_shortcut, shortcut_type, conv, norm):
    shortcut = x
    shape_in = x.get_shape()
    with tf.variable_scope(name, use_resource=True):
        with tf.variable_scope("1", use_resource=True):
            x = conv(x, 3, stride, filters)
            x = norm['one_init'](x)
            x = tf.nn.relu(x)

        with tf.variable_scope("2", use_resource=True):
            x = conv(x, 3, 1, filters)
            x = norm['one_init'](x)

        with tf.variable_scope("p", use_resource=True):
            if use_shortcut:
                if shortcut_type == "A":
                    pad = int(x.get_shape()[3] - shape_in[3])
                    if stride != 1:
                        shortcut = tf.strided_slice(
                            shortcut,
                            [0, 0, 0, 0],
                            shortcut.get_shape(),
                            strides=[1, stride, stride, 1],
                        )
                    shortcut = tf.pad(
                        shortcut, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]]
                    )
                else:  # shortcut_type B
                    shortcut = conv(shortcut, 1, stride, filters)
                    shortcut = norm['one_init'](shortcut)
        x = shortcut + x
    return x


# Shortcut types:
# A: Zero padding when increasing dims
# B: 1x1 conv when increasing dims
# C: 1x1 conv on all shortcuts
def block3(x, stride, filters, name, use_shortcut, shortcut_type, norm, conv,
           conv_groups=1, filter_factor=1):
    shortcut = x
    split_filters = filters[0] * filter_factor
    with tf.variable_scope(name, use_resource=True):
        with tf.variable_scope("1", use_resource=True):
            x = conv(x, 1, 1, split_filters)
            x = norm['one_init'](x)
            x = tf.nn.relu(x)

        with tf.variable_scope("2", use_resource=True):
            x = conv(x, 3, stride, split_filters, groups=conv_groups)
            x = norm['one_init'](x)
            x = tf.nn.relu(x)

        with tf.variable_scope("3", use_resource=True):
            x = conv(x, 1, 1, filters[1])
            if use_shortcut:
                x = norm['one_init'](x)
            else:
                x = norm['zero_init'](x)

        with tf.variable_scope("p", use_resource=True):
            if use_shortcut:
                shortcut = conv(shortcut, 1, stride, filters[1])
                shortcut = norm['one_init'](shortcut)

        x = shortcut + x
    return x


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == "channels_first":
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        )
    else:
        padded_inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )
    return padded_inputs


ResNetDefinition = namedtuple(
    "ResNetDefinition",
    ["initial_block_filters", "block_fn", "block_counts", "out_filters"],
)

RESNETS_Bottleneck_Cifar_wide = {
    9 * n + 2: ResNetDefinition(
        64, block3, [n, n, n], [(64, 256), (128, 512), (256, 1024)]
    )
    for n in [1, 3, 4, 5, 6, 7]
}

RESNETS_Bottleneck_Cifar = {
    9 * n + 2: ResNetDefinition(
        16, block3, [n, n, n], [(16, 64), (32, 128), (64, 256)]
    )
    for n in [1, 3, 4, 5, 6, 7]
}

RESNETS_Cifar = {
    6 * n + 2: ResNetDefinition(16, block2, [n, n, n], [16, 32, 64])
    for n in [1, 2, 3, 5, 7, 9, 18]
}

RESNETS_Bottleneck_Imagenet = {
    14: ResNetDefinition(64, block3,
                         [1, 1, 1, 1],
                         [(64, 256), (128, 512), (256, 1024), (512, 2048)]),
    26: ResNetDefinition(64, block3,
                         [2, 2, 2, 2],
                         [(64, 256), (128, 512), (256, 1024), (512, 2048)]),
    50: ResNetDefinition(64, block3,
                         [3, 4, 6, 3],
                         [(64, 256), (128, 512), (256, 1024), (512, 2048)]),
    101: ResNetDefinition(64, block3,
                          [3, 4, 23, 3],
                          [(64, 256), (128, 512), (256, 1024), (512, 2048)]),
    152: ResNetDefinition(64, block3,
                          [3, 8, 36, 3],
                          [(64, 256), (128, 512), (256, 1024), (512, 2048)])
}

RESNETS_Imagenet = {
    10: ResNetDefinition(64, block2, [1, 1, 1, 1], [64, 128, 256, 512]),
    18: ResNetDefinition(64, block2, [2, 2, 2, 2], [64, 128, 256, 512]),
    34: ResNetDefinition(64, block2, [3, 4, 6, 3], [64, 128, 256, 512]),
}


def add_resnet_base_arguments(group):
    group.add_argument("--model-size", type=int, help="Size of the model.")
    group.add_argument("--batch-norm", action="store_true",
                       help="Use batch norm (CIFAR Default)")
    group.add_argument("--group-norm", action="store_true",
                       help="Use group norm (ImageNet Default)")
    group.add_argument("--groups", type=int, help="Number of groups")
    group.add_argument("--BN-decay", type=float,
                       help="Decay (or momentum) used for the BN weighted " +
                       "mean and variance.")
