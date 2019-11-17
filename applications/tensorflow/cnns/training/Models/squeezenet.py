# Copyright 2019 Graphcore Ltd.
"""
SqueezeNet

A Convolutional ineural network with relatively few parameters (~1.25M) but equivalent
accuracy to AlexNet.

Architecture originally described in Caffe. Implemented here in Tensorflow for the IPU.


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
from functools import partial
import base as BASE
import validation as VALID
from tensorflow.contrib.ipu.python.poprand import dropout


class SqueezeNet:
    def __init__(self, opts, is_training=True):
        self.is_training = is_training
        self.num_classes = 1000

    def _build_graph(self, image):
        """Classifies a batch of ImageNet images

        Returns:
        A logits Tensor with shape [<batch_size>, self.num_classes]
        """
        image = _conv1(image, name="initialconv")
        x = tf.compat.v1.layers.max_pooling2d(image, pool_size=3, strides=2)
        x = _fire(x, 16, 64, 64, name="fire2")
        x = _fire(x, 16, 64, 64, name="fire3")
        x = _fire(x, 32, 128, 128, name="fire4")
        # maxpool4
        x = tf.compat.v1.layers.max_pooling2d(x, pool_size=3, strides=2)
        x = _fire(x, 32, 128, 128, name="fire5")
        x = _fire(x, 48, 192, 192, name="fire6")
        x = _fire(x, 48, 192, 192, name="fire7")
        x = _fire(x, 64, 256, 256, name="fire8")
        # maxpool8
        x = tf.compat.v1.layers.max_pooling2d(x, pool_size=3, strides=2)
        x = _fire(x, 64, 256, 256, name="fire9")
        x = tf.nn.dropout(
            x, keep_prob=0.5 if self.is_training else 0.0, name="drop9")
        image = _conv10(x, name="finalconv")
        avgpool = tf.layers.average_pooling2d(
            image, pool_size=13, strides=1, name="final_pool")
        logits = tf.layers.flatten(avgpool)
        return logits

    def __call__(self, x):
        shape = x.get_shape().as_list()
        if len(shape) != 4:
            raise ValueError("Input size must be [batch,height,width,channels]")
        if shape[1] < 224 or shape[2] < 224:
            raise ValueError("Input image must be at least 224x224")
        return self._build_graph(x)


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


def _conv10(inputs, name):
    """The first layer of squeezenet
    Convolution then average pooling

    name: a string name for the tensor output of the block layer.

    Returns:
        The output tensor of the block.
    """
    with tf.variable_scope(name):
        inputs = conv(inputs, ksize=1, stride=1, filters_out=1000,
                      kernel_initializer=tf.initializers.truncated_normal(
                          mean=0.0, stddev=0.01),
                      bias=False)
        return inputs


def _fire(inputs, s_1, e_1, e_3, name):
    """Fire module:
    A 'squeeze' convolution layer, which has only 1x1 filters, feeding
    into an 'expand' layer, that has a mix of 1x1 and 3x3 filters.

    s_1: The number of 1x1 filters in the squeeze layer
    e_1: The number of 1x1 filters in the expand layer
    e_3: The number of 3x3 filters in the expand layer
    name: a string name for the tensor output of the block layer.
    """
    with tf.variable_scope(name):
        # squeeze layer
        with tf.variable_scope(name+"s_1"):
            inputs = conv(inputs, ksize=1, stride=1, filters_out=s_1,
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                          bias=False)
            s1_out = tf.nn.relu(inputs)
        # expand layer
        with tf.variable_scope(name+"e_1"):
            e1_out = conv(s1_out, ksize=1, stride=1, filters_out=e_1,
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                          bias=False)
            e1_out = tf.nn.relu(e1_out)
        # expand layer
        with tf.variable_scope(name+"e_3"):
            e3_out = conv(s1_out, ksize=3, stride=1, filters_out=e_3,
                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                          bias=False)
            e3_out = tf.nn.relu(e3_out)
        inputs = tf.concat([e1_out, e3_out], axis=3, name='concat')
        return inputs


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
    return parser


def set_defaults(opts):
    opts['summary_str'] += "SqueezeNet\n"

    if opts['dataset'] == 'imagenet':
        opts['shortcut_type'] = 'B'
    elif 'cifar' in opts['dataset']:
        opts['shortcut_type'] = 'A'

#    opts['dataset'] = 'imagenet'
    opts['lr_schedule'] = 'polynomial_decay_lr'

    if not opts.get('epochs') and not opts.get('iterations'):
        opts['epochs'] = 100

    if not opts.get("batch_size"):
        opts['batch_size'] = 4

    if (opts['precision'] == '32.32') and not opts.get("shards"):
        opts['shards'] = 2

    opts['name'] = "SN_bs{}".format(opts['batch_size'])

    if opts.get('replicas') > 1:
        opts['name'] += "x{}r".format(opts['replicas'])
    if opts['pipeline_depth'] > 1:
        opts['name'] += "x{}p".format(opts['pipeline_depth'])
    elif opts.get('gradients_to_accumulate') > 1:
        opts['name'] += "x{}a".format(opts['gradients_to_accumulate'])

    opts['name'] += '_{}{}'.format(opts['precision'],
                                   '_noSR' if opts['no_stochastic_rounding'] else '')
