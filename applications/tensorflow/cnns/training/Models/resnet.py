# Copyright 2019 Graphcore Ltd.
import tensorflow as tf
from collections import namedtuple
from functools import partial
from tensorflow.python.ipu import normalization_ops


def max_pool(x, ksize=3, stride=2):
    x = tf.nn.max_pool(
        x,
        ksize=[1, ksize, ksize, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')
    tf.add_to_collection('activations', x)
    return x


def norm(x, opts, is_training=True):
    norm_type = 'GROUP' if opts["group_norm"] else 'BATCH' if opts['batch_norm'] else None

    if norm_type == 'BATCH':
        x = tf.layers.batch_normalization(x, fused=True, center=True, scale=True,
                                          training=is_training, trainable=True,
                                          momentum=opts["BN_decay"], epsilon=1e-5)
    elif norm_type == 'GROUP':
        x = normalization_ops.group_norm(x, groups=opts['groups'])

    tf.add_to_collection('activations', x)
    return x


def fc(x, num_units_out, name, seed=None):
    with tf.variable_scope(name, use_resource=True):
        x = tf.layers.dense(inputs=x, units=num_units_out,
                            kernel_initializer=tf.glorot_uniform_initializer(seed=seed))
        tf.add_to_collection('activations', x)
        return x


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Further details of this is necessary can be found at:
    https://www.tensorflow.org/versions/r1.8/api_guides/python/nn#Convolution
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv(x, ksize, stride, filters_out, bias=True, seed=None):
    with tf.variable_scope('conv', use_resource=True):
        if stride > 1:
            x = fixed_padding(x, ksize, 'channels_last')
        return tf.layers.conv2d(
            inputs=x, filters=filters_out, kernel_size=ksize, strides=stride,
            padding=('SAME' if stride == 1 else 'VALID'), use_bias=bias,
            kernel_initializer=tf.variance_scaling_initializer(seed=seed),
            data_format='channels_last')


def initial_block(x, filters, ksize=3, initial_downsample=False, conv=conv, norm=norm, name="b0"):
    with tf.variable_scope(name, use_resource=True):
        x = conv(x, ksize, 2 if initial_downsample else 1, filters)
        x = norm(x)
        x = tf.nn.relu(x)
        if initial_downsample:
            x = max_pool(x, ksize=3, stride=2)
        return x


# Shortcut types:
# A: Zero padding when increasing dims
# B: 1x1 conv when increasing dims
# C: 1x1 conv on all shortcuts
def block2(x, stride, filters, name, use_shortcut, shortcut_type, conv=conv, norm=norm):
    shortcut = x
    shape_in = x.get_shape()
    with tf.variable_scope(name, use_resource=True):
        with tf.variable_scope("1", use_resource=True):
            x = conv(x, 3, stride, filters)
            x = norm(x)
            x = tf.nn.relu(x)

        with tf.variable_scope("2", use_resource=True):
            x = conv(x, 3, 1, filters)
            x = norm(x)

        with tf.variable_scope("p", use_resource=True):
            if use_shortcut:
                if shortcut_type == 'A':
                    pad = int(x.get_shape()[3] - shape_in[3])
                    if stride != 1:
                        shortcut = tf.strided_slice(shortcut, [0, 0, 0, 0], shortcut.get_shape(),
                                                    strides=[1, stride, stride, 1])
                    shortcut = tf.pad(shortcut, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]])
                else:  # shortcut_type B
                    shortcut = conv(shortcut, 1, stride, filters)
                    shortcut = norm(shortcut)
        x = shortcut + x
    return x


def block3(x, stride, filters, name, use_shortcut, shortcut_type, conv=conv, norm=norm):
    shortcut = x
    shape_in = x.get_shape()
    with tf.variable_scope(name, use_resource=True):
        with tf.variable_scope("1", use_resource=True):
            x = conv(x, 1, 1, filters[0])
            x = norm(x)
            x = tf.nn.relu(x)

        with tf.variable_scope("2", use_resource=True):
            x = conv(x, 3, stride, filters[0])
            x = norm(x)
            x = tf.nn.relu(x)

        with tf.variable_scope("3", use_resource=True):
            x = conv(x, 1, 1, filters[1])
            x = norm(x)

        with tf.variable_scope("p", use_resource=True):
            if use_shortcut:
                shortcut = conv(shortcut, 1, stride, filters[1])
                shortcut = norm(shortcut)

        x = shortcut + x
    return x


def final_block_relu(x, name):
    with tf.variable_scope(name, use_resource=True):
        return tf.nn.relu(x)


ResNetDefinition = namedtuple(
    'ResNetDefinition', ['initial_block_fn',
                         'block_fn',
                         'block_counts',
                         'out_filters'])


RESNETS = {
    # Cifar10 sized filters
    8:  ResNetDefinition(initial_block, block2,
                         [1, 1, 1],
                         [16, 16, 32, 64]),
    14:  ResNetDefinition(initial_block, block2,
                          [2, 2, 2],
                          [16, 16, 32, 64]),
    20:  ResNetDefinition(initial_block, block2,
                          [3, 3, 3],
                          [16, 16, 32, 64]),
    32:  ResNetDefinition(initial_block, block2,
                          [5, 5, 5],
                          [16, 16, 32, 64]),
    44:  ResNetDefinition(initial_block, block2,
                          [7, 7, 7],
                          [16, 16, 32, 64]),
    56:  ResNetDefinition(initial_block, block2,
                          [9, 9, 9],
                          [16, 16, 32, 64]),
    110: ResNetDefinition(initial_block, block2,
                          [18, 18, 18],
                          [16, 16, 32, 64]),
    # ImageNet sized filters
    10: ResNetDefinition(initial_block, block2,
                         [1, 1, 1, 1],
                         [64, 64, 128, 256, 512]),
    18: ResNetDefinition(initial_block, block2,
                         [2, 2, 2, 2],
                         [64, 64, 128, 256, 512]),
    34: ResNetDefinition(initial_block, block2,
                         [3, 4, 6, 3],
                         [64, 64, 128, 256, 512]),
    50: ResNetDefinition(initial_block, block3,
                         [3, 4, 6, 3],
                         [64, (64, 256), (128, 512), (256, 1024), (512, 2048)]),
    101: ResNetDefinition(initial_block, block3,
                          [3, 4, 23, 3],
                          [64, (64, 256), (128, 512), (256, 1024), (512, 2048)])
}


def custom_dtype_getter(getter, name, dtype, trainable, master_weight_filter_fn, shape=None,
                        *args, **kwargs):
    master_dtype = master_weight_filter_fn(name)
    if dtype != master_dtype and trainable:
        var = getter(name, shape, master_dtype, *args, trainable=trainable, **kwargs)
        return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
        return getter(name, shape, dtype, *args, trainable=trainable, **kwargs)


class ResNet:
    def __init__(self, opts, is_training=True):
        dtypes = opts["precision"].split('.')
        self.dtype = tf.float16 if dtypes[0] == '16' else tf.float32

        self.master_weight_filter_fn = lambda name: tf.float32 \
            if dtypes[1] == '32' else tf.float16

        self.custom_dtype_getter = partial(custom_dtype_getter,
                                           master_weight_filter_fn=self.master_weight_filter_fn)
        definition = RESNETS[opts["model_size"]]

        # Apply options to layers
        self.conv = partial(conv,
                            bias=not (opts["batch_norm"] or opts["group_norm"]),
                            seed=opts["seed"])

        self.norm = partial(norm,
                            opts=opts,
                            is_training=is_training)
        self.fc = partial(fc,
                          seed=opts["seed"])

        # Apply changed layers to block functions
        self.initial_block_fn = partial(definition.initial_block_fn,
                                        conv=self.conv,
                                        norm=self.norm)

        self.block_fn = partial(definition.block_fn,
                                shortcut_type=opts["shortcut_type"],
                                conv=self.conv,
                                norm=self.norm)

        self.block_counts = definition.block_counts
        self.out_filters = definition.out_filters

        self.out_filters = [int(f) if not isinstance(f, tuple)
                            else tuple(int(f2) for f2 in f)
                            for f in self.out_filters]

        # Apply dataset specific changes
        if opts["dataset"] == 'imagenet':
            self.num_classes = 1000
            self.initial_block_fn = partial(self.initial_block_fn,
                                            ksize=7,
                                            initial_downsample=True)
        elif opts["dataset"] == 'cifar-10':
            self.num_classes = 10
            self.initial_block_fn = partial(self.initial_block_fn,
                                            ksize=3,
                                            initial_downsample=False)
        elif opts["dataset"] == 'cifar-100':
            self.num_classes = 100
            self.initial_block_fn = partial(self.initial_block_fn,
                                            ksize=3,
                                            initial_downsample=False)
        else:
            raise ValueError("Unknown Dataset {}".format(opts["dataset"]))

    def _build_function_list(self):
        fn_list = []
        fn_list.append(partial(self.initial_block_fn, filters=self.out_filters[0], name='b0'))

        for n in range(len(self.block_counts)):
            first_stride = 1 if n is 0 else 2
            for i in range(self.block_counts[n]):
                stride = (first_stride if (i == 0) else 1)
                fn_list.append(partial(self.block_fn, stride=stride, filters=self.out_filters[n+1],
                                       name="b{}/{}".format(n+1, i), use_shortcut=(i == 0)))
                fn_list.append(partial(final_block_relu, name="b{}/{}/relu".format(n+1, i)))
        fn_list.append(partial(tf.reduce_mean, reduction_indices=[1, 2], name="reduce_mean"))
        fn_list.append(partial(self.fc, num_units_out=self.num_classes, name="fc1"))
        return fn_list

    def build_whole_graph(self, x):
        fn_list = self._build_function_list()

        tf.add_to_collection('activations', x)
        with tf.variable_scope('all', use_resource=True, custom_getter=self.custom_dtype_getter):
            for fn in fn_list:
                x = fn(x)
        return x

    def first_stage(self, x, first_split_name):
        self.fn_list = self._build_function_list()
        if first_split_name not in [fn.keywords['name'] for fn in self.fn_list]:
            raise ValueError("Couldn't find pipeline split called " + first_split_name)
        tf.add_to_collection('activations', x)
        with tf.variable_scope('all', use_resource=True, custom_getter=self.custom_dtype_getter):
            for fn in self.fn_list:
                if fn.keywords['name'] == first_split_name:
                    break
                x = fn(x)
        return x

    def later_stage(self, x, prev_split_name, end_split_name):
        if end_split_name is not None and end_split_name not in [fn.keywords['name'] for fn in self.fn_list]:
            raise ValueError("Couldn't find pipeline split called " + end_split_name)
        with tf.variable_scope('all', use_resource=True, custom_getter=self.custom_dtype_getter):
            got_first_stage = False
            for fn in self.fn_list:
                if not got_first_stage and fn.keywords['name'] != prev_split_name:
                    continue
                got_first_stage = True
                if fn.keywords['name'] == end_split_name:
                    break
                x = fn(x)
        return x

    def __call__(self, x):
        return self.build_whole_graph(x)


def Model(opts, training, image):
    return ResNet(opts, training)(image)


def staged_model(opts):
    splits = opts['pipeline_splits']
    x = ResNet(opts, True)
    if splits is None:
        possible_splits = [s.keywords['name'] for s in x._build_function_list() if 'relu' in s.keywords['name']]
        raise ValueError("--pipeline-splits not specified. Need {} of {}".format(opts['shards'] - 1, possible_splits))
    splits.append(None)
    stages = [partial(x.first_stage, first_split_name=splits[0])]
    for i in range(len(splits) - 1):
        stages.append(partial(x.later_stage, prev_split_name=splits[i], end_split_name=splits[i + 1]))
    return stages


def add_arguments(parser):
    group = parser.add_argument_group('ResNet')
    group.add_argument('--model-size', type=int,
                       help='Size of the ResNet.')
    group.add_argument('--batch-norm', action="store_true",
                       help="Use batch norm (CIFAR Default)")
    group.add_argument('--group-norm', action="store_true",
                       help="Use group norm (ImageNet Default)")
    group.add_argument('--groups', type=int,
                       help="Number of groups")
    group.add_argument('--BN-decay', type=float,
                       help="Decay (or momentum) used for the BN weighted mean and variance.")
    return parser


def set_defaults(opts):
    opts['summary_str'] += "ResNet-{model_size}\n"

    if opts['dataset'] == 'imagenet':
        opts['shortcut_type'] = 'B'
    elif 'cifar' in opts['dataset']:
        opts['shortcut_type'] = 'A'

    # set ImageNet specific defaults
    if opts['dataset'] == 'imagenet':
        if not opts.get('weight_decay'):
            # value taken from tf_official_resnet - may not be appropriate for small batch sizes
            opts['weight_decay'] = 1e-4
        if not opts.get('base_learning_rate'):
            if opts['optimiser'] == 'SGD':
                opts['base_learning_rate'] = -8
            elif opts['optimiser'] == 'momentum':
                opts['base_learning_rate'] = -11
        if not opts.get('epochs') and not opts.get('iterations'):
            opts['epochs'] = 100
        if not opts.get('learning_rate_schedule'):
            opts['learning_rate_schedule'] = [0.3, 0.6, 0.8, 0.9]
        if not opts.get('learning_rate_decay'):
            opts['learning_rate_decay'] = [1.0, 0.1, 0.01, 0.001, 1e-4]
        if not (opts.get("group_norm") is True or opts.get("batch_norm") is True):
            # set group norm as default for ImageNet
            opts['group_norm'] = True
        if opts.get("group_norm"):
            if not opts.get("groups"):
                opts['groups'] = 32
        if not opts.get("model_size"):
            opts['model_size'] = 18
        if not opts.get("batch_size"):
            opts['batch_size'] = 4
        if opts.get("warmup") is None:
            # warmup on by default for ImageNet
            opts['warmup'] = True

        # exclude beta and gamma from weight decay calculation
        opts["wd_exclude"] = ['beta', 'gamma']

    # set CIFAR specific defaults
    elif 'cifar' in opts['dataset']:
        if not opts.get('weight_decay'):
            # based on sweep with CIFAR-10
            opts['weight_decay'] = 1e-6
        if not opts.get('base_learning_rate'):
            opts['base_learning_rate'] = -6
        if not opts.get('epochs') and not opts.get('iterations'):
            opts['epochs'] = 160
        if not opts.get('learning_rate_schedule'):
            opts['learning_rate_schedule'] = [0.5, 0.75]
        if not opts.get('learning_rate_decay'):
            opts['learning_rate_decay'] = [1.0, 0.1, 0.01]
        if not (opts.get("group_norm") is True or opts.get("batch_norm") is True):
            # set batch norm as default for CIFAR
            opts['batch_norm'] = True
        if opts.get("group_norm"):
            if not opts.get("groups"):
                opts['groups'] = 16
        if not opts.get("model_size"):
            opts['model_size'] = 20
        if not opts.get("batch_size"):
            opts['batch_size'] = 32

    if not opts['BN_decay']:
        opts['BN_decay'] = 0.97

    opts['name'] = "RN{}".format(opts['model_size'])

    opts['name'] += "_bs{}".format(opts['batch_size'])
    if opts.get('replicas') > 1:
        opts['name'] += "x{}r".format(opts['replicas'])
    if opts['pipeline_depth'] > 1:
        opts['name'] += "x{}p".format(opts['pipeline_depth'])
    elif opts.get('gradients_to_accumulate') > 1:
        opts['name'] += "x{}a".format(opts['gradients_to_accumulate'])

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
        if opts['BN_decay'] and opts['BN_decay'] != 0.97:  # defined and not default
            opts['summary_str'] += "  Decay: {}\n".format(opts['BN_decay'])

    opts['name'] += '_{}{}'.format(opts['precision'], '_noSR' if opts['no_stochastic_rounding'] else '')
