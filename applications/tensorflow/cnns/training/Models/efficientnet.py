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
import tensorflow as tf
from collections import namedtuple
from functools import partial
from .model_base import ModelBase
import collections
import math
from six.moves import xrange
import string
from tensorflow.python.ipu import normalization_ops
from tensorflow.python.ipu import nn_ops
from tensorflow.python.keras import layers
from .batch_norm import batch_norm
from .proxy_norm import proxynorm_activation


BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def get_default_block_args(expand_ratio=6, se_ratio=0.25, cifar=False):
    return [
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
                  expand_ratio=1, id_skip=True, strides=1, se_ratio=se_ratio),
        BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
                  expand_ratio=expand_ratio, id_skip=True, strides=1 if cifar else 2, se_ratio=se_ratio),
        BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
                  expand_ratio=expand_ratio, id_skip=True, strides=2, se_ratio=se_ratio),
        BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
                  expand_ratio=expand_ratio, id_skip=True, strides=2, se_ratio=se_ratio),
        BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
                  expand_ratio=expand_ratio, id_skip=True, strides=1, se_ratio=se_ratio),
        BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
                  expand_ratio=expand_ratio, id_skip=True, strides=2, se_ratio=se_ratio),
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
                  expand_ratio=expand_ratio, id_skip=True, strides=1, se_ratio=se_ratio)
    ]


# Define the coefficients for width, depth, and dropout rate
EfficientNetsDefinition = namedtuple(
    'EfficientNetsDefinition', ['width_coefficient',
                                'depth_coefficient',
                                'dropout_rate'])
EFFICIENTNET = {
    'CIFAR': EfficientNetsDefinition(0.5, 1.0, 0),
    'B0': EfficientNetsDefinition(1.0, 1.0, 0.2),
    'B1': EfficientNetsDefinition(1.0, 1.1, 0.2),
    'B2': EfficientNetsDefinition(1.1, 1.2, 0.3),
    'B3': EfficientNetsDefinition(1.2, 1.4, 0.3),
    'B4': EfficientNetsDefinition(1.4, 1.8, 0.4),
    'B5': EfficientNetsDefinition(1.6, 2.2, 0.4),
    'B6': EfficientNetsDefinition(1.8, 2.6, 0.5),
    'B7': EfficientNetsDefinition(2.0, 3.1, 0.5)
    }

EFFICIENTNET_DEFAULT_SIZE = {
    'CIFAR': 32,
    'B0': 224,
    'B1': 240,
    'B2': 260,
    'B3': 300,
    'B4': 380,
    'B5': 456,
    'B6': 528,
    'B7': 600
    }


def fc(x, num_units_out, name, seed=None):
    with tf.variable_scope(name, use_resource=True):
        x = tf.layers.dense(inputs=x, units=num_units_out,
                            kernel_initializer=tf.glorot_uniform_initializer(seed=seed))
        return x


def conv2d_basic(x, ksize, stride, filters_out, bias=False, groups=None, group_dim=None,
                 kernel_initializer=None, seed=None, name='conv'):
    if kernel_initializer is None:
        kernel_initializer = tf.variance_scaling_initializer(seed=seed)
    with tf.variable_scope(name, use_resource=True):
        in_filters = x.get_shape().as_list()[3]
        assert (groups is None) or (group_dim is None)
        if groups is None:
            if group_dim is None:
                groups = 1
            else:
                groups = max(in_filters // group_dim, 1)

        W = tf.get_variable("conv2d/kernel",
                            shape=[ksize, ksize, in_filters//groups, filters_out],
                            dtype=x.dtype,
                            trainable=True,
                            initializer=kernel_initializer)
        x = tf.nn.conv2d(x,
                         filters=W,
                         strides=[1, stride, stride, 1],
                         padding='SAME')
        if bias:
            b = tf.get_variable("conv2d/bias",
                                shape=[filters_out],
                                dtype=x.dtype,
                                trainable=True,
                                initializer=tf.zeros_initializer())
            x = tf.nn.bias_add(x, b)
        tf.add_to_collection('activations', x)
        return x


def norm(x, is_training=True, norm_type=None, center=True, scale=True,
         epsilon=1e-6, groups=None):
    if norm_type == 'layer':
        norm_type = 'group'
        groups = 1

    if norm_type == 'batch':
        x = batch_norm(x, center=center, scale=scale, training=is_training,
                       trainable=True, epsilon=epsilon)
    elif norm_type == 'group':
        x = normalization_ops.group_norm(x, center=center, scale=scale,
                                         groups=min(int(x.get_shape()[3]), groups),
                                         epsilon=epsilon)
    else:
        raise ValueError("{} not a valid norm_type".format(norm_type))

    return x


def round_filters(filters, width_coefficient, depth_divisor):
    """
        Round number of filters based on width multiplier.
    """
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """
        Round number of repeats based on depth multiplier.
    """
    return int(math.ceil(depth_coefficient * repeats))


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = tf.div(inputs, survival_prob) * binary_tensor
    return output


def mb_conv_block_funs(
                  in_filters,
                  out_filters,
                  kernel_size=3,
                  stride=1,
                  expand_ratio=6,
                  se_ratio=0.0,
                  survival_prob=1.0,
                  is_training=True,
                  id_skip=False,
                  proxy_norm=False,
                  activation=None,
                  conv=conv2d_basic,
                  norm=None,
                  depthwise_conv=None,
                  name=''):

    intermediate_filters = in_filters * expand_ratio
    id_skip = id_skip and stride == 1 and in_filters == out_filters
    has_se = (se_ratio is not None) and (0 < se_ratio <= 1)

    def mb_conv_a(input, name=name):
        with tf.variable_scope(name + "/a"):
            x = conv(input, 1, 1, intermediate_filters)
            x = norm(x)
            x = activation(x)
            return x

    def mb_conv_b(input, has_se=True, name=name + "/b"):
        with tf.variable_scope(name):
            # Depthwise Convolution
            x = depthwise_conv(input, kernel_size, stride, filters_out=intermediate_filters)
            x = norm(x)
            if proxy_norm:
                x, inv_proxy_std = activation(x, delay_scale=True)  # delay proxy scaling for efficiency
            else:
                x = activation(x)

        if has_se:
            with tf.variable_scope(name[:-1] + "SE"):
                num_reduced_filters = max(1, int(in_filters * se_ratio))
                se_tensor = tf.reduce_mean(x, reduction_indices=[1, 2], keep_dims=True)
                with tf.variable_scope("1"):
                    se_tensor = conv(se_tensor, 1, 1, num_reduced_filters, bias=True)
                if proxy_norm:
                    se_tensor = activation(se_tensor, proxy_norm=False)
                else:
                    se_tensor = activation(se_tensor)
                with tf.variable_scope("2"):
                    se_tensor = conv(se_tensor, 1, 1, intermediate_filters, bias=True)
                se_tensor = tf.math.sigmoid(se_tensor)
                if proxy_norm:
                    se_tensor = se_tensor * inv_proxy_std  # do proxy scaling here for efficiency
                x = tf.math.multiply(x, se_tensor)

        return x

    def mb_conv_c(input, name=name + "/c"):
        with tf.variable_scope(name):
            x = conv(input, 1, 1, out_filters)
            # we cannot use proxy norm here as this is not followed by an activation
            # we find that disabling the shift parameter is a good alternative to reducing the denormalisation
            x = norm(x, scale=True)
        return x

    def full_block(x_in, name=name):
        x = x_in
        if expand_ratio != 1:
            x = mb_conv_a(x)
        x = mb_conv_b(x, has_se=has_se)
        x = mb_conv_c(x)
        if survival_prob > 0 and survival_prob < 1:
            x = drop_connect(x, is_training, survival_prob)
        x = tf.math.add(x, x_in)
        return x

    """Mobile Inverted Residual Bottleneck."""
    fn_list = []
    # Currently the pipelining functions throw an error for outputs that are lists
    if id_skip:
        fn_list += [partial(full_block, name=name)]
    else:
        if expand_ratio != 1:
            fn_list += [partial(mb_conv_a, name=name)]
        fn_list += [partial(mb_conv_b, has_se=has_se, name=name + "/b")]
        fn_list += [partial(mb_conv_c, name=name + "/c")]

    return fn_list


class EfficientNet(ModelBase):
    def __init__(self, opts, is_training=True):
        super(EfficientNet, self).__init__(opts)
        self.is_training = is_training

        # Block defintion
        self.model_size = opts['model_size']
        definition = EFFICIENTNET[self.model_size]

        # width_coefficient: float, scaling coefficient for network width.
        if opts['width_coefficient']:
            self.width_coefficient = opts['width_coefficient']
        else:
            self.width_coefficient = definition.width_coefficient

        # depth_coefficient: float, scaling coefficient for network depth.
        if opts['depth_coefficient']:
            self.depth_coefficient = opts['depth_coefficient']
        else:
            self.depth_coefficient = definition.depth_coefficient

        # dropout_rate: float, dropout rate before final classifier layer.
        if opts['dropout_rate'] is not None:
            self.dropout_rate = opts['dropout_rate']
        else:
            self.dropout_rate = definition.dropout_rate

        self.block_survival_prob = opts['block_survival_prob']

        # blocks_args: A list of BlockArgs to construct block modules.
        self.blocks_args = get_default_block_args(expand_ratio=opts['expand_ratio'],
                                                  se_ratio=0 if opts['no_se'] else 0.25,
                                                  cifar=self.model_size == 'CIFAR')
        # require explicit definition of depth-divisor (default=8)
        self.depth_divisor = opts['depth_divisor']

        # Activation
        if opts["use_relu"]:
            self.activation = tf.nn.relu
        else:
            try:
                self.activation = nn_ops.swish
            except AttributeError:
                self.activation = tf.nn.swish
                print("IPU nn_ops.swish operation not found. Falling back to tf.nn.swish .")

        self.proxy_norm = opts['proxy_norm']

        if self.proxy_norm:
            self.activation = partial(proxynorm_activation,
                                      activation=self.activation,
                                      proxy_epsilon=opts['proxy_epsilon'],
                                      proxy_recompute=opts['proxy_recompute'])

        # Standard building block layers
        self.top_width = round_filters(opts['top_width'], self.width_coefficient, self.depth_divisor)

        # 2D convolution
        self.conv = partial(conv2d_basic,
                            kernel_initializer=tf.variance_scaling_initializer(scale=2, mode='fan_out'))

        # Depthwise separable
        self.group_conv = partial(conv2d_basic,
                                  bias=False,
                                  kernel_initializer=tf.variance_scaling_initializer(scale=2, mode='fan_in'),
                                  group_dim=opts['group_dim'])

        # Fully-connected
        self.fc = fc
        # Normalization
        self.norm = partial(norm,
                            is_training=is_training,
                            norm_type=opts['norm_type'],
                            groups=opts["groups"],
                            epsilon=opts["norm_epsilon"],
                            center=not self.proxy_norm,
                            scale=not self.proxy_norm)
        # Apply changed layers to block functions
        self.mb_conv_block_fns = partial(mb_conv_block_funs,
                                         conv=self.conv,
                                         norm=self.norm,
                                         depthwise_conv=self.group_conv,
                                         activation=self.activation,
                                         proxy_norm=self.proxy_norm)

    def block0(self, x, stride=2, name="block0"):
        with tf.variable_scope(name, use_resource=True):
            x = self.conv(x, 3, stride, round_filters(32, self.width_coefficient, self.depth_divisor), bias=False)
            x = self.norm(x)
            x = self.activation(x)
            return x

    def top(self, x, name="Top"):
        with tf.variable_scope(name, use_resource=True):
            if self.top_width > 0:
                x = self.conv(x, 1, 1, self.top_width, bias=False)
                x = self.norm(x)
                if self.proxy_norm:
                    x = self.activation(x, delay_scale=True)[0]
                else:
                    x = self.activation(x)
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            if self.dropout_rate > 0:
                x = tf.layers.Dropout(self.dropout_rate, name='top_dropout')(x, training=self.is_training)
            x = self.fc(x, self.num_classes, 'logits', seed=None)
            return x

    def _build_function_list(self):
        fn_list = []
        fn_list.append(partial(self.block0, stride=1 if self.model_size == 'CIFAR' else 2,
                               name="block0"))

        # Build blocks
        num_blocks_total = sum(round_repeats(block_args.num_repeat, self.depth_coefficient) for block_args in self.blocks_args)
        block_num = 0
        for idx, block_args in enumerate(self.blocks_args):
            assert block_args.num_repeat > 0
            input_filters = round_filters(block_args.input_filters,
                                          self.width_coefficient,
                                          self.depth_divisor)
            output_filters = round_filters(block_args.output_filters,
                                           self.width_coefficient,
                                           self.depth_divisor)
            num_repeat = round_repeats(block_args.num_repeat, self.depth_coefficient)

            for bidx in range(num_repeat):
                drop_connect_rate = 1 - self.block_survival_prob
                drop_connect_rate *= float(block_num / num_blocks_total)
                survival_prob = 1 - drop_connect_rate

                block_name = 'block{}{}'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx]
                )
                fn_list += self.mb_conv_block_fns(
                                       in_filters=input_filters,
                                       out_filters=output_filters,
                                       kernel_size=block_args.kernel_size,
                                       stride=1 if bidx > 0 else block_args.strides,
                                       expand_ratio=block_args.expand_ratio,
                                       se_ratio=block_args.se_ratio,
                                       survival_prob=survival_prob,
                                       is_training=self.is_training,
                                       id_skip=block_args.id_skip,
                                       name=block_name)
                input_filters = output_filters
                block_num += 1

        fn_list.append(partial(self.top, name="Top"))

        return fn_list


def Model(opts, training, image):
    return EfficientNet(opts, training)(image)


def staged_model(opts):
    splits = opts['pipeline_splits']
    x = EfficientNet(opts, True)
    if splits is None or (
            len(splits) != opts['shards'] - 1 and opts['shards'] > 1):
        possible_splits = [
            s.keywords['name'] for s in x._build_function_list()
        ]
        raise ValueError(
            "--pipeline-splits not specified or wrong number of splits. Need {} of {}".format(
                opts['shards'] - 1, possible_splits))
    splits.append(None)
    stages = [partial(x.first_stage, first_split_name=splits[0])]
    for i in range(len(splits) - 1):
        stages.append(
            partial(x.later_stage,
                    prev_split_name=splits[i],
                    end_split_name=splits[i + 1]))
    return stages


def add_arguments(parser):
    group = parser.add_argument_group('EfficientNet')
    group.add_argument('--model-size', type=str.upper, default='B0',
                       help='Size of EfficientNet (default="B0")')
    group.add_argument('--dropout-rate', type=float,
                       help='Override default dropout rate')
    group.add_argument('--block-survival-prob', type=float, default=0.8,
                       help='Override default drop connect survival probability')
    group.add_argument('--no-se', action="store_true",
                       help='Disable Squeeze and Excite blocks')
    group.add_argument('--depth-divisor', type=int, default=8,
                       help='Round channel widths to this factor')
    group.add_argument('--top-width', type=int, default=1280,
                       help='Number of channels for final 1x1')
    group.add_argument('--use-relu', action="store_true",
                       help='Use ReLU instead of default swish activation')
    group.add_argument('--expand-ratio', type=int, default=6,
                       help='Expansion ratio')
    group.add_argument('--group-dim', type=int, default=1,
                       help='Dimension of group convs (default=1 i.e. depthwise convs)')
    group.add_argument('--BN-decay', type=float, default=0.97,
                       help="Decay (or momentum) used for the BN weighted mean and variance.")
    group.add_argument("--batch-norm", action="store_true",
                       help="Use batch norm")
    group.add_argument("--group-norm", action="store_true",
                       help="Use group norm")
    group.add_argument('--groups', type=int, default=4,
                       help="Number of groups for group norm")
    group.add_argument('--norm-epsilon', type=float, default=1e-3,
                       help="Epsilon used for normalization")

    group.add_argument('--proxy-norm', action='store_true', default=False,
                       help="Use proxy norm")
    group.add_argument('--proxy-epsilon', type=float, default=0.03,
                       help="Numerical stability constant for proxy scaling")
    group.add_argument('--no-proxy-recompute', action='store_false', dest='proxy_recompute',
                       help="Dont recompute cheap proxy norm ops")
    group.set_defaults(proxy_recompute=True)

    group.add_argument('--width-coefficient', type=float,
                       help='Override width coefficient')
    group.add_argument('--depth-coefficient', type=float,
                       help='Override width coefficient')
    return parser


def set_defaults(opts):
    try:
        ms = int(opts['model_size'])
        opts['model_size'] = f"B{ms}"
    except:
        pass
    # set ImageNet specific defaults
    if opts['dataset'] == 'imagenet':
        if opts.get('image_size') is None:
            opts['image_size'] = EFFICIENTNET_DEFAULT_SIZE[opts['model_size']]
        if opts.get("weight_decay") is None:
            wd_default = 0 if opts.get("optimiser") == "LARS" else 1e-5
            opts["weight_decay"] = wd_default
        if opts.get("lars_weight_decay") is None:
            opts["lars_weight_decay"] = 1e-5
        if not opts.get('base_learning_rate_exponent'):
            if opts['optimiser'] == 'SGD':
                opts['base_learning_rate_exponent'] = -7.0
            elif opts['optimiser'] == 'momentum':
                opts['base_learning_rate_exponent'] = -10.0
            elif opts['optimiser'] == 'RMSprop':
                opts['base_learning_rate_exponent'] = -14.0
        if opts['optimiser'] == 'RMSprop' and opts['rmsprop_base_decay_exp'] is None:
            opts['rmsprop_base_decay_exp'] = -14.0
        if not opts.get('epochs') and not opts.get('iterations'):
            opts['epochs'] = 350.0
        if opts.get('lr_schedule') == 'stepped':
            if not opts.get('learning_rate_schedule'):
                opts['learning_rate_schedule'] = [0.3, 0.6, 0.8, 0.9]
            if not opts.get('learning_rate_decay'):
                opts['learning_rate_decay'] = [1.0, 0.1, 0.01, 0.001, 1e-4]
        elif opts.get('lr_schedule') == 'exponential':
            if not opts.get('lr_decay_rate'):
                opts['lr_decay_rate'] = 0.97
            if not opts.get('lr_drops'):
                opts['lr_drops'] = 146
        if not opts.get("batch_size"):
            opts['batch_size'] = 4
        if opts.get("warmup") is None:
            opts['warmup'] = True
        if not opts.get("label_smoothing"):
            opts['label_smoothing'] = 0.1
        # force stable norm on
        if not opts.get("stable_norm"):
            opts['stable_norm'] = True

        # exclude beta and gamma from weight decay calculation
        opts["wd_exclude"] = ['beta', 'gamma', 'proxy', 'moving_']
    elif 'cifar' in opts['dataset']:
        if not opts.get("model_size"):
            opts['model_size'] = 'cifar'
        if opts.get('image_size') is None:
            opts['image_size'] = EFFICIENTNET_DEFAULT_SIZE[opts['model_size']]
        if opts.get("weight_decay") is None:
            wd_default = 0 if opts.get("optimiser") == "LARS" else 1e-6
            opts["weight_decay"] = wd_default
        if opts.get("lars_weight_decay") is None:
            opts["lars_weight_decay"] = 1e-6
        if not opts.get('base_learning_rate_exponent'):
            opts['base_learning_rate_exponent'] = -6
        if not opts.get('epochs') and not opts.get('iterations'):
            opts['epochs'] = 300
        if opts.get('lr_schedule') == 'stepped':
            if not opts.get('learning_rate_schedule'):
                opts['learning_rate_schedule'] = [0.5, 0.75]
            if not opts.get('learning_rate_decay'):
                opts['learning_rate_decay'] = [1.0, 0.1, 0.01]
        elif opts.get('lr_schedule') == 'exponential':
            if not opts.get('lr_decay_rate'):
                opts['lr_decay_rate'] = 0.97
            if not opts.get('lr_drops'):
                opts['lr_drops'] = 146
        if not opts.get("batch_size"):
            opts['batch_size'] = 8
    else:
        raise ValueError('Only the ImageNet and CIFAR datasets are currently supported for EfficientNet.')

    if not (opts.get("group_norm") is True or opts.get("batch_norm") is True):
        # set group norm as default
        opts['group_norm'] = True
    if not opts.get("group_norm"):
        opts['groups'] = 1

    opts['norm_type'] = 'batch' if opts.get('batch_norm') else ('group' if opts.get('group_norm') else 'none')

    opts['summary_str'] += "EfficientNet-{model_size}\n"

    opts['name'] = "EfficientNet-{}".format(opts['model_size'])

    opts['name'] += "_bs{}".format(opts['batch_size'])
    if opts['pipeline']:
        opts['name'] += "x{}p".format(opts['gradient_accumulation_count'])
    elif opts.get('gradient_accumulation_count') > 1:
        opts['name'] += "x{}a".format(opts['gradient_accumulation_count'])

    if opts.get('replicas') > 1:
        opts['name'] += "x{}r".format(opts['replicas'])

    if not (opts["batch_norm"] or opts['group_norm']):
        opts['name'] += '_noBN'
        opts['summary_str'] += " No Batch Norm\n"
    elif opts["group_norm"]:
        opts['name'] += '_GN{}'.format(opts['groups'])
        opts['summary_str'] += (" Group Norm\n"
                                "  {groups} groups\n")
    else:
        opts['name'] += '_BN'
        opts['summary_str'] += " Batch Norm\n"
        opts['summary_str'] += "  Decay: {}\n".format(opts['BN_decay'])
