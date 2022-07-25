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

import sys
sys.path.append("..")
import log as logging


class ResNetBase(ModelBase):
    """Base model for ResNet and ResNeXt"""

    def __init__(self, opts, definition, is_training=True):
        super().__init__(opts, is_training)

        # Apply options to layers
        self.conv = partial(conv, seed=opts["seed"])

        norm_name = (
            "gn" if opts["group_norm"] else
            "bn" if opts["batch_norm"] else
            "norm")
        self.norm_dict = {
            "norm_name": norm_name,
            "one_init": partial(
                norm, opts=opts, is_training=is_training),
            "zero_init": partial(
                norm, opts=opts, is_training=is_training, zero_gamma_init = True)}
        self.fc = partial(fc, seed=opts["seed"])

        # Apply changed layers to block functions
        self.initial_block_fn = partial(
            initial_block, conv=self.conv, norm=self.norm_dict)

        self.initial_block_filters = definition.initial_block_filters
        self.block_counts = definition.block_counts
        self.out_filters = definition.out_filters

        self.out_filters = [
            int(f) if not isinstance(f, tuple) else tuple(int(f2) for f2 in f)
            for f in self.out_filters]

        self.block_fn = partial(definition.block_fn, shortcut_type=opts["shortcut_type"],
                                conv=self.conv, norm=self.norm_dict)

        # Apply dataset specific changes
        if opts["dataset"] == "imagenet":
            self.initial_block_fn = partial(
                self.initial_block_fn, ksize=7, initial_downsample=True)
        elif opts["dataset"] == "cifar-10":
            self.initial_block_fn = partial(
                self.initial_block_fn, ksize=3, initial_downsample=False)
        elif opts["dataset"] == "cifar-100":
            self.initial_block_fn = partial(
                self.initial_block_fn, ksize=3, initial_downsample=False)

    def _build_function_list(self):
        fn_list = []
        fn_list.append(
            partial(self.initial_block_fn, filters=self.initial_block_filters, name="stage0"))

        for n in range(len(self.block_counts)):
            first_stride = 1 if n is 0 else 2
            for i in range(self.block_counts[n]):
                stride = first_stride if (i == 0) else 1
                fn_list.append(
                    partial(self.block_fn, stride=stride, filters=self.out_filters[n],
                            name="stage{}/unit{}".format(n + 1, i + 1), use_shortcut=(i == 0)))
                fn_list.append(
                    partial(final_block_relu, name="stage{}/unit{}/relu".format(n + 1, i + 1)))
        fn_list.append(
            partial(tf.reduce_mean, reduction_indices=[1, 2], name="reduce_mean"))
        fn_list.append(
            partial(self.fc, num_units_out=self.num_classes, name="fc1"))
        return fn_list


class MLPerfInitializerWrapper():
    """Wrapper class for logging variable initialization."""
    popdist_instance = None
    validation_phase = False

    def __init__(self, tf_initializer, tensor_name, stack_offset,
                 instance_to_log=0, mlperf_tensor_name_format=True):
        """
        Args:
            tf_initializer: TensorFlow initializer callable.
            tensor_name: Tensor name.
            stack_offset: Stack offset depth for logging.
            instance_to_log: If `None` will log all variable initializations.
            mlperf_tensor_name_format: If `True`, uses '_' instead of '/' with
            no parent scope: /resnet_50/a/b/c/weight -> a_b_c_weight
        """
        current_variable_scope_name = tf.get_variable_scope().name
        self._tf_initializer = tf_initializer
        self._tensor_name = "/".join([current_variable_scope_name, tensor_name])
        if mlperf_tensor_name_format:
            self._tensor_name = "_".join(self._tensor_name.split("/")[1:])
            self._tensor_name = self._tensor_name.replace("stage0_", "")
        self._stack_offset = stack_offset
        self._instance_to_log = instance_to_log

        # Conversion to mllog.constants is done in logging.mlperf_logging!
        self._event_name = "WEIGHTS_INITIALIZATION"

        if self.popdist_instance is None:
            self._instance_rank = 0
            self._total_number_of_instances = 1
        else:
            self._instance_rank = self.popdist_instance.getInstanceIndex()
            self._total_number_of_instances = self.popdist_instance.getNumTotalReplicas()

    def _log_init_call(self,):
        if self._instance_to_log >= self._total_number_of_instances:
            raise ValueError(
                f"Instance index, {self._instance_to_log}, cannot be larger than"
                f" total number of instances ({self._total_number_of_instances})!")

        if self._instance_to_log == self._instance_rank and not self.validation_phase:
            logging.mlperf_logging(
                key=self._event_name, value=None, log_type="event",
                metadata=dict(tensor=self._tensor_name),
                stack_offset=self._stack_offset)

    def __call__(self, *args, **kwargs):
        self._log_init_call()
        return self._tf_initializer.__call__(*args, **kwargs)


def norm(x, opts, is_training=True, zero_gamma_init=False):
    # We initialise the parameters depending on the gamma_init argument
    if zero_gamma_init:
        p_init = {
            "gamma": MLPerfInitializerWrapper(
                tf_initializer=tf.zeros_initializer(),
                tensor_name="gamma",
                stack_offset=27,     # 19 for Python, 26 for PopRun
                instance_to_log=0),
            "beta": MLPerfInitializerWrapper(
                tf_initializer=tf.zeros_initializer(),
                tensor_name="beta",
                stack_offset=27,     # 19 for Python, 26 for PopRun
                instance_to_log=0)}
    else:
        p_init = {
            "gamma": MLPerfInitializerWrapper(
                tf_initializer=tf.ones_initializer(),
                tensor_name="gamma",
                stack_offset=27,     # 19 for Python, 26 for PopRun
                instance_to_log=0),
            "beta": MLPerfInitializerWrapper(
                tf_initializer=tf.zeros_initializer(),
                tensor_name="beta",
                stack_offset=27,     # 19 for Python, 26 for PopRun
                instance_to_log=0)}

    norm_type = (
        "GROUP"
        if opts["group_norm"]
        else "BATCH"
        if opts["batch_norm"]
        else None)

    if norm_type == "BATCH":
        x = batch_norm(
            x,
            center=True,
            scale=True,
            training=is_training,
            trainable=True,
            epsilon=1e-5,
            gamma_initializer=p_init["gamma"],
            beta_initializer=p_init["beta"])
    elif norm_type == "GROUP":
        # Adding arguments to group norm
        x = normalization_ops.group_norm(
            x,
            groups=opts["groups"],
            param_initializers=p_init)

    tf.add_to_collection("activations", x)
    return x


def fc(x, num_units_out, name, seed=None):
    with tf.variable_scope(name, use_resource=True):
        last_dim = x.get_shape().as_list()[-1]
        weight = tf.get_variable(
            "weight",
            shape=[last_dim, num_units_out],
            dtype=x.dtype,
            trainable=True,
            initializer=MLPerfInitializerWrapper(
                tf_initializer=tf.glorot_uniform_initializer(seed=seed),
                tensor_name="weight",
                stack_offset=26,     # 18 for Python, 26 for PopRun
                instance_to_log=0))
        bias = tf.get_variable(
            "bias",
            shape=[num_units_out, ],
            dtype=x.dtype,
            trainable=True,
            initializer=MLPerfInitializerWrapper(
                tf_initializer=tf.zeros_initializer(),
                tensor_name="bias",
                stack_offset=26,     # 18 for Python, 26 for PopRun
                instance_to_log=0))
        x = tf.linalg.matmul(x, weight)
        x = tf.nn.bias_add(x, bias)
        tf.add_to_collection("activations", x)
        return x


def max_pool(x, ksize=3, stride=2):
    x = tf.nn.max_pool(
        x,
        ksize=[1, ksize, ksize, 1],
        strides=[1, stride, stride, 1],
        padding="SAME",)
    tf.add_to_collection("activations", x)
    return x


def conv(x, ksize, stride, filters_out, groups=1, seed=None):
    in_filters = x.get_shape().as_list()[3]
    weight = tf.get_variable(
        "weight",
        shape=[ksize, ksize, in_filters / groups, filters_out],
        dtype=x.dtype,
        trainable=True,
        initializer=MLPerfInitializerWrapper(
                tf_initializer=tf.variance_scaling_initializer(seed=seed),
                tensor_name="weight",
                stack_offset=26,     # 18 for Python, 26 for PopRun
                instance_to_log=0))
    return tf.nn.conv2d(
        x,
        filters=weight,
        strides=[1, stride, stride, 1],
        padding="SAME",
        data_format="NHWC")


def initial_block(x, filters, conv, norm, ksize=3, initial_downsample=False, name="b0"):
    norm_name = norm["norm_name"]
    with tf.variable_scope(name, use_resource=True):
        with tf.variable_scope("conv0", use_resource=True):
            x = conv(x, ksize, 2 if initial_downsample else 1, filters)
        with tf.variable_scope(f"{norm_name}0", use_resource=True):
            x = norm["one_init"](x)
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
    norm_name = norm["norm_name"]
    with tf.variable_scope(name, use_resource=True):
        with tf.variable_scope("conv1", use_resource=True):
            x = conv(x, 3, stride, filters)
        with tf.variable_scope(f"{norm_name}1", use_resource=True):
            x = norm["one_init"](x)
        x = tf.nn.relu(x)

        with tf.variable_scope("conv2", use_resource=True):
            x = conv(x, 3, 1, filters)
        with tf.variable_scope(f"{norm_name}2", use_resource=True):
            x = norm["one_init"](x)

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
                with tf.variable_scope("conv1sc", use_resource=True):
                    shortcut = conv(shortcut, 1, stride, filters)
                with tf.variable_scope(f"{norm_name}sc", use_resource=True):
                    shortcut = norm["one_init"](shortcut)
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
    norm_name = norm["norm_name"]
    with tf.variable_scope(name, use_resource=True):
        with tf.variable_scope("conv1", use_resource=True):
            x = conv(x, 1, 1, split_filters)
        with tf.variable_scope(f"{norm_name}1", use_resource=True):
            x = norm["one_init"](x)
        x = tf.nn.relu(x)

        with tf.variable_scope("conv2", use_resource=True):
            x = conv(x, 3, stride, split_filters, groups=conv_groups)
        with tf.variable_scope(f"{norm_name}2", use_resource=True):
            x = norm["one_init"](x)
        x = tf.nn.relu(x)

        with tf.variable_scope("conv3", use_resource=True):
            x = conv(x, 1, 1, filters[1])
        with tf.variable_scope(f"{norm_name}3", use_resource=True):
            if use_shortcut:
                x = norm["one_init"](x)
            else:
                x = norm["zero_init"](x)

        if use_shortcut:
            with tf.variable_scope("conv1sc", use_resource=True):
                shortcut = conv(shortcut, 1, stride, filters[1])
            with tf.variable_scope(f"{norm_name}sc", use_resource=True):
                shortcut = norm["one_init"](shortcut)

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
        64, block3, [n, n, n], [(64, 256), (128, 512), (256, 1024)])
    for n in [1, 3, 4, 5, 6, 7]
}

RESNETS_Bottleneck_Cifar = {
    9 * n + 2: ResNetDefinition(
        16, block3, [n, n, n], [(16, 64), (32, 128), (64, 256)])
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
