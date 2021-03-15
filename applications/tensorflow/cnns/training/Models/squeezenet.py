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

"""
SqueezeNet

A Convolutional Neural Network with relatively few parameters (~1.25M) but equivalent
accuracy to AlexNet.

Architecture originally described in Caffe. Implemented here in TensorFlow for the IPU.


SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH
50X FEWER PARAMETERS AND <0.5MB MODEL SIZE
https://arxiv.org/pdf/1602.07360.pdf

SqueezeNet was originally implemented using a polynomial decay learnng rate. To use this,
run with --lr-schedule polynomial_decay_lr.

This will set a default rate of linear decay of the learning rate. You can change this
rate with the parameters --poly-lr-decay-steps, --poly-lr-initial-lr, --poly-lr-decay-power
and --poly-lr-end-lr.

Unlike the original implementation, this version does not use quantization or compression.
Additionally, unlike the original, this model trains in fp16 as default, but can be run in
fp32 with --precision 32.32. In this case, you will need to run over two IPUs.
"""

import tensorflow as tf


class SqueezeNet:
    def __init__(self, opts, is_training=True):
        self.is_training = is_training
        # Apply dataset specific changes
        if opts["dataset"] == "imagenet":
            self.num_classes = 1000
        elif opts["dataset"] == "cifar-10":
            self.num_classes = 10
        elif opts["dataset"] == "cifar-100":
            self.num_classes = 100
        else:
            raise ValueError("Unknown Dataset {}".format(opts["dataset"]))
        self.use_bypass = opts.get("use_bypass")

    def _build_graph(self, image, use_bypass):
        """Classifies a batch of ImageNet images

        Returns:
        A logits Tensor with shape [<batch_size>, self.num_classes]
        """
        pool_size, strides = (3, 2)
        image = _conv1(image, name="initialconv")
        x = tf.compat.v1.layers.max_pooling2d(image, pool_size=pool_size, strides=strides)
        x = _fire(x, 16, 64, name="fire2")
        x = _fire(x, 16, 64, name="fire3", use_bypass=use_bypass)
        x = _fire(x, 32, 128, name="fire4")
        # maxpool4
        x = tf.compat.v1.layers.max_pooling2d(x, pool_size=pool_size, strides=strides)
        x = _fire(x, 32, 128, name="fire5", use_bypass=use_bypass)
        x = _fire(x, 48, 192, name="fire6")
        x = _fire(x, 48, 192, name="fire7", use_bypass=use_bypass)
        x = _fire(x, 64, 256, name="fire8")
        # maxpool8
        x = tf.compat.v1.layers.max_pooling2d(x, pool_size=pool_size, strides=strides)
        x = _fire(x, 64, 256, name="fire9", use_bypass=use_bypass)
        x = tf.nn.dropout(
            x, rate=0.5 if self.is_training else 0.0, name="drop9")
        if self.num_classes == 1000:
            x = _conv10(x, name="finalconv", num_classes=self.num_classes)
            x = tf.layers.average_pooling2d(
                x, pool_size=13, strides=1, name="final_pool")
            x = tf.layers.flatten(x)
        else:
            x = tf.layers.flatten(x)
            x = tf.layers.dense(units=self.num_classes*2, inputs=x, activation=tf.nn.relu)
        return x

    def __call__(self, x):
        shape = x.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input size must be [batch,height,width,channels]")
        return self._build_graph(x, self.use_bypass)


def Model(opts, training, image):
    return SqueezeNet(opts, training)(image)


###########################################
# SqueezeNet block definitions
###########################################

def _conv1(inputs, name):
    """The first layer of squeezenet
    Convolution

    name: a string name for the tensor output of the block layer.

    Returns:
        The output tensor of the block.
    """
    with tf.variable_scope(name):
        inputs = conv(inputs, ksize=7, stride=2, filters_out=96,
                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), bias=False)
        return inputs


def _conv10(inputs, name, num_classes):
    """The first layer of squeezenet
    Convolution then average pooling

    name: a string name for the tensor output of the block layer.

    Returns:
        The output tensor of the block.
    """
    with tf.variable_scope(name):
        init = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        inputs = conv(inputs, ksize=1, stride=1, filters_out=num_classes,
                      kernel_initializer=init, bias=False)
        return inputs


def _fire(inputs, squeeze, expand, name, use_bypass=False):
    """Fire module:
    A 'squeeze' convolution layer, which has only 1x1 filters, feeding
    into an 'expand' layer, that has a mix of 1x1 and 3x3 filters.

    squeeze: The number of 1x1 filters in the squeeze layer
    expand: The number of 1x1 and 3x3 filters in the expand layer
    name: a string name for the tensor output of the block layer.
    """
    with tf.variable_scope(name):
        # squeeze layer
        with tf.variable_scope(name+"s_1"):
            x = conv(inputs, ksize=1, stride=1, filters_out=squeeze,
                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                     bias=False)
            s1_out = tf.nn.relu(x)
        # expand layer
        with tf.variable_scope(name+"e_1"):
            e1_out = conv(s1_out, ksize=1, stride=1, filters_out=expand,
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                          bias=False)
            e1_out = tf.nn.relu(e1_out)
        # expand layer
        with tf.variable_scope(name+"e_3"):
            e3_out = conv(s1_out, ksize=3, stride=1, filters_out=expand,
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                          bias=False)
            e3_out = tf.nn.relu(e3_out)
        x = tf.concat([e1_out, e3_out], axis=3, name='concat')
        if use_bypass:
            x = inputs + x
        return x


###########################################
# Layer definitions
###########################################

def conv(x, ksize, stride, filters_out, kernel_initializer, bias=True):
    with tf.variable_scope('conv', use_resource=True):

        return tf.layers.conv2d(
            inputs=x, filters=filters_out, kernel_size=ksize, strides=stride,
            padding='same',
            use_bias=bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
            activation=tf.nn.relu,
            data_format='channels_last')


def add_arguments(parser):
    group = parser.add_argument_group('SqueezeNet')
    group.add_argument('--use-bypass', action='store_true', help="Use bypass in the fire module.")
    return parser


def set_defaults(opts):
    opts['summary_str'] += "SqueezeNet\n"

    if not opts.get("lr_schedule"):
        opts['lr_schedule'] = 'polynomial_decay_lr'

    # set ImageNet specific defaults
    if opts["dataset"] == "imagenet":
        if opts.get("weight_decay") is None:
            # value taken from tf_official_resnet - may not be appropriate for
            # small batch sizes
            wd_default = 0 if opts.get("optimiser") == "LARS" else 1e-4
            opts["weight_decay"] = wd_default
        if opts.get("lars_weight_decay") is None:
            opts["lars_weight_decay"] = 1e-4
        if not opts.get("base_learning_rate"):
            if opts["optimiser"] == "SGD":
                opts["base_learning_rate"] = -8
            elif opts["optimiser"] == "momentum":
                opts["base_learning_rate"] = -11
        if not opts.get("learning_rate_schedule"):
            opts["learning_rate_schedule"] = [0.3, 0.6, 0.8, 0.9]
        if not opts.get("learning_rate_decay"):
            opts["learning_rate_decay"] = [1.0, 0.1, 0.01, 0.001, 1e-4]
        if opts.get("warmup") is None:
            # warmup on by default for ImageNet
            opts["warmup"] = True
        if not opts.get("batch_size"):
            opts['batch_size'] = 2
        # exclude beta and gamma from weight decay calculation
        opts["wd_exclude"] = ["beta", "gamma"]

    # set CIFAR specific defaults
    elif "cifar" in opts["dataset"]:
        if opts.get("weight_decay") is None:
            # based on sweep with CIFAR-10
            wd_default = 0 if opts.get("optimiser") == "LARS" else 1e-6
            opts["weight_decay"] = wd_default
        if opts.get("lars_weight_decay") is None:
            opts["lars_weight_decay"] = 1e-6
        if not opts.get("base_learning_rate"):
            opts["base_learning_rate"] = -10
        if not opts.get("epochs") and not opts.get("iterations"):
            opts["epochs"] = 160
        if not opts.get("learning_rate_schedule"):
            opts["learning_rate_schedule"] = [0.5, 0.75]
        if not opts.get("learning_rate_decay"):
            opts["learning_rate_decay"] = [1.0, 0.1, 0.01]
        if not opts.get("batch_size"):
            opts['batch_size'] = 128

    if not opts.get('epochs') and not opts.get('iterations'):
        opts['epochs'] = 100

    if (opts['precision'] == '32.32') and not opts.get("shards"):
        opts['shards'] = 2

    opts['name'] = "SN_bs{}".format(opts['batch_size'])

    if opts.get('replicas') > 1:
        opts['name'] += "x{}r".format(opts['replicas'])
    if opts['pipeline']:
        opts['name'] += "x{}p".format(opts['gradient_accumulation_count'])
    elif opts.get('gradient_accumulation_count') > 1:
        opts['name'] += "x{}a".format(opts['gradient_accumulation_count'])

    opts['name'] += '_{}{}'.format(opts['precision'],
                                   '_noSR' if opts['no_stochastic_rounding'] else '')
