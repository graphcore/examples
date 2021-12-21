# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# ------------------------------------------------------------
#     The Resnet base backbone on IPU with Popart
#     Paper reference:https://arxiv.org/abs/1512.03385
#     More info is at:https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN
#     Written by Sicong L, modified by Hu Di
# ------------------------------------------------------------

from IPU.ipu_tensor import gcop
from models.base_model import BaseModel
from utils import logger
from config import cfg


def my_conv(x,
            ksize,
            stride,
            pads,
            c_out,
            bias=True,
            group=1,
            training=True,
            bias_training=None,
            fp16_on=False,
            weights_fp16_on=None,
            debugPrefix=''):
    assert pads == ksize // 2
    assert group == 1
    if isinstance(stride, int):
        stride = [stride] * 2

    result = gcop.cF.conv2d(x,
                            c_out,
                            ksize=ksize,
                            bias=bias,
                            train=training,
                            strides=stride,
                            padding_mode='same',
                            debugContext=debugPrefix,
                            fp16_on=fp16_on,
                            weights_fp16_on=weights_fp16_on,
                            bias_training=bias_training)
    return result


class Resnet(BaseModel):
    def __init__(
        self,
        classes=[1] * 1000,
        network_type='50',
        fp16_on=False,
        training=False,
        fix_bn=False,
        fix_blocks=1,
    ):

        super().__init__(fp16_on=fp16_on, training=training)
        self.classes = classes
        self.classes_n = len(classes)
        self.network_type = network_type
        self.bn_training = not fix_bn
        self.fix_blocks = fix_blocks
        if training:
            self.fc_fp16 = cfg.TRAIN.FC_FP16
        else:
            self.fc_fp16 = cfg.TEST.FC_FP16

        logger.log_str('network_type:', self.network_type)
        if self.network_type == '50':
            self.layers_count = [3, 4, 6, 3]
            self.layers_out = [256, 512, 1024, 2048]
            self.layers_stride = cfg.MODEL.LAYERS_STRIDE
        elif self.network_type == '18':
            self.layers_count = [2, 2, 2, 2]
            self.layers_out = [64, 128, 256, 512]
            self.layers_stride = cfg.MODEL.LAYERS_STRIDE
        else:
            raise NotImplementedError
        self.total_stride = 4
        for s in cfg.MODEL.LAYERS_STRIDE:
            self.total_stride = self.total_stride * s

    def conv3x3(
        self,
        x,
        stride=1,
        pads=1,
        c_out=64,
        debugPrefix='conv3x3',
        bias=False,
        group=1,
        batch_norm=False,
        weights_fp16_on=None,
        training=True,
        bias_training=None,
    ):

        x = my_conv(x,
                    3,
                    stride,
                    pads,
                    c_out,
                    bias,
                    group,
                    training=training,
                    fp16_on=None,
                    weights_fp16_on=weights_fp16_on,
                    bias_training=bias_training,
                    debugPrefix=debugPrefix)
        if batch_norm:
            with gcop.variable_scope('bn'):
                local_bn_training = self.bn_training if self.training else False
                x = gcop.layers.batch_normalization(
                    x, axis=1, fp16_on=None, trainable=local_bn_training)
        return x

    def conv1x1(self,
                x,
                c_out,
                debugPrefix='conv1x1',
                batch_norm=False,
                stride=1,
                bias=False,
                training=True,
                weights_fp16_on=None,
                bias_training=None):
        x = my_conv(x,
                    1,
                    stride,
                    0,
                    c_out,
                    bias=bias,
                    training=training,
                    fp16_on=None,
                    weights_fp16_on=weights_fp16_on,
                    bias_training=bias_training,
                    debugPrefix=debugPrefix)
        if batch_norm:
            with gcop.variable_scope('bn'):
                local_bn_training = self.bn_training if self.training else False
                x = gcop.layers.batch_normalization(
                    x, axis=1, fp16_on=None, trainable=local_bn_training)

        return x

    def downsample(self, x, stride=2, training=True, weights_fp16_on=None):
        B, C, H, W = x.shape.as_list()
        x = my_conv(x,
                    1,
                    stride,
                    0,
                    C,
                    bias=False,
                    fp16_on=None,
                    weights_fp16_on=weights_fp16_on,
                    training=training)
        local_bn_training = self.bn_training if self.training else False
        x = gcop.layers.batch_normalization(x,
                                            axis=1,
                                            fp16_on=None,
                                            trainable=local_bn_training)
        return x

    def residual(self,
                 x,
                 shortcut,
                 c_out,
                 stride,
                 training=True,
                 weights_fp16_on=None,
                 bias_training=None):
        #
        B, C, H, W = x.shape.as_list()
        B1, C1, H1, W1 = shortcut.shape.as_list()
        conv_bias = False
        if C != C1 or stride != 1:
            shortcut = my_conv(shortcut,
                               1,
                               stride,
                               0,
                               c_out,
                               bias=conv_bias,
                               fp16_on=None,
                               training=training,
                               weights_fp16_on=weights_fp16_on,
                               bias_training=bias_training,
                               debugPrefix='0')
            local_bn_training = self.bn_training if self.training else False
            shortcut = gcop.layers.batch_normalization(
                shortcut,
                axis=1,
                trainable=local_bn_training,
                fp16_on=None,
                name='1')
        x = shortcut + x
        x = gcop.nn.relu(x)
        return x

    def bottleneck_block(
        self,
        x,
        stride,
        pads=1,
        c_out=64,
        expansion=4,
        count=None,
        debugPrefix='',
        training=True,
    ):
        local_bn_training = self.bn_training if training else False
        conv_bias = False
        for i in range(count):
            shortcut = x

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(
                    x,
                    c_out=c_out // expansion,
                    stride=stride if i == 0 else 1,
                    bias=conv_bias,
                    debugPrefix='conv1',
                    training=training,
                    bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn1')
                x = gcop.nn.relu(x)

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv3x3(x,
                                 c_out=c_out // expansion,
                                 bias=conv_bias,
                                 debugPrefix='conv2',
                                 training=training,
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn2')
                x = gcop.nn.relu(x)

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(x,
                                 c_out=c_out,
                                 bias=conv_bias,
                                 debugPrefix='conv3',
                                 training=training,
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn3')

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i)) +
                                     "/downsample"):
                x = self.residual(x,
                                  shortcut=shortcut,
                                  c_out=c_out,
                                  stride=stride if i == 0 else 1,
                                  training=training,
                                  bias_training=local_bn_training)
        return x

    def bottleneck_block_single(
        self,
        x,
        stride,
        pads=1,
        c_out=64,
        expansion=4,
        i=None,
        debugPrefix='',
        training=True,
    ):
        conv_bias = False
        lastlayer_first_stride = 1
        lastlayer_second_stride = stride

        shortcut = x

        local_bn_training = self.bn_training if training else False
        with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
            x = self.conv1x1(x,
                             c_out=c_out // expansion,
                             stride=lastlayer_first_stride,
                             bias=conv_bias,
                             debugPrefix='conv1',
                             training=training,
                             bias_training=local_bn_training)
            x = gcop.layers.batch_normalization(x,
                                                axis=1,
                                                trainable=local_bn_training,
                                                fp16_on=None,
                                                name='bn1')
            x = gcop.nn.relu(x)

        with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
            x = self.conv3x3(x,
                             stride=lastlayer_second_stride,
                             c_out=c_out // expansion,
                             bias=conv_bias,
                             debugPrefix='conv2',
                             training=training,
                             bias_training=local_bn_training)
            x = gcop.layers.batch_normalization(x,
                                                axis=1,
                                                trainable=local_bn_training,
                                                fp16_on=None,
                                                name='bn2')
            x = gcop.nn.relu(x)

        with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
            x = self.conv1x1(x,
                             c_out=c_out,
                             bias=conv_bias,
                             debugPrefix='conv3',
                             training=training,
                             bias_training=local_bn_training)
            x = gcop.layers.batch_normalization(x,
                                                axis=1,
                                                trainable=local_bn_training,
                                                fp16_on=None,
                                                name='bn3')

        with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i)) +
                                 "/downsample"):
            x = self.residual(x,
                              shortcut=shortcut,
                              c_out=c_out,
                              stride=stride if i == 0 else 1,
                              training=training,
                              bias_training=local_bn_training)

        return x

    def head_to_tail(self, x, ipu_configs):
        conv_weights_fp16_on_l = [None for i in range(12)]
        for _idx in cfg.MODEL.RCNN.CONV_WEIGHTS_FP16_OFF_INDICES:
            conv_weights_fp16_on_l[_idx] = False
        with gcop.device(ipu_configs[6]):
            x, stride, pads, c_out, expansion, i, debugPrefix, training = x, self.layers_stride[
                3], 1, self.layers_out[3], 4, 0, 'layer4', self.training
            conv_bias = False

            if cfg.MODEL.LAYER4S:
                lastlayer_first_stride = stride
                lastlayer_second_stride = 1
            else:
                lastlayer_first_stride = 1
                lastlayer_second_stride = stride

            shortcut = x

            local_bn_training = self.bn_training if training else False
            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(x,
                                 c_out=c_out // expansion,
                                 stride=lastlayer_first_stride,
                                 bias=conv_bias,
                                 debugPrefix='conv1',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[0],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn1')
                x = gcop.nn.relu(x)
        with gcop.device(ipu_configs[7]):
            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv3x3(x,
                                 stride=lastlayer_second_stride,
                                 c_out=c_out // expansion,
                                 bias=conv_bias,
                                 debugPrefix='conv2',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[1],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn2')
                x = gcop.nn.relu(x)

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(x,
                                 c_out=c_out,
                                 bias=conv_bias,
                                 debugPrefix='conv3',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[2],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn3')

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i)) +
                                     "/downsample"):
                x = self.residual(x,
                                  shortcut=shortcut,
                                  c_out=c_out,
                                  stride=stride if i == 0 else 1,
                                  training=training,
                                  weights_fp16_on=conv_weights_fp16_on_l[3],
                                  bias_training=local_bn_training)

        with gcop.device(ipu_configs[8]):
            x, stride, pads, c_out, expansion, i, debugPrefix, training = x, 1, 1, self.layers_out[
                3], 4, 1, 'layer4', self.training
            conv_bias = False
            lastlayer_first_stride = 1
            lastlayer_second_stride = stride

            shortcut = x

            local_bn_training = self.bn_training if training else False
            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(x,
                                 c_out=c_out // expansion,
                                 stride=lastlayer_first_stride,
                                 bias=conv_bias,
                                 debugPrefix='conv1',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[4],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn1')
                x = gcop.nn.relu(x)

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv3x3(x,
                                 stride=lastlayer_second_stride,
                                 c_out=c_out // expansion,
                                 bias=conv_bias,
                                 debugPrefix='conv2',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[5],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn2')
                x = gcop.nn.relu(x)

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(x,
                                 c_out=c_out,
                                 bias=conv_bias,
                                 debugPrefix='conv3',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[6],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn3')

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i)) +
                                     "/downsample"):
                x = self.residual(x,
                                  shortcut=shortcut,
                                  c_out=c_out,
                                  stride=stride if i == 0 else 1,
                                  training=training,
                                  weights_fp16_on=conv_weights_fp16_on_l[7],
                                  bias_training=local_bn_training)

        with gcop.device(ipu_configs[9]):
            x, stride, pads, c_out, expansion, i, debugPrefix, training = x, 1, 1, self.layers_out[
                3], 4, 2, 'layer4', self.training
            conv_bias = False
            lastlayer_first_stride = 1
            lastlayer_second_stride = stride

            shortcut = x

            local_bn_training = self.bn_training if training else False
            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(x,
                                 c_out=c_out // expansion,
                                 stride=lastlayer_first_stride,
                                 bias=conv_bias,
                                 debugPrefix='conv1',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[8],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn1')
                x = gcop.nn.relu(x)
        with gcop.device(ipu_configs[10]):
            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv3x3(x,
                                 stride=lastlayer_second_stride,
                                 c_out=c_out // expansion,
                                 bias=conv_bias,
                                 debugPrefix='conv2',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[9],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn2')
                x = gcop.nn.relu(x)

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i))):
                x = self.conv1x1(x,
                                 c_out=c_out,
                                 bias=conv_bias,
                                 debugPrefix='conv3',
                                 training=training,
                                 weights_fp16_on=conv_weights_fp16_on_l[10],
                                 bias_training=local_bn_training)
                x = gcop.layers.batch_normalization(
                    x,
                    axis=1,
                    trainable=local_bn_training,
                    fp16_on=None,
                    name='bn3')

            with gcop.variable_scope(debugPrefix + "/" + '{}'.format(str(i)) +
                                     "/downsample"):
                x = self.residual(x,
                                  shortcut=shortcut,
                                  c_out=c_out,
                                  stride=stride if i == 0 else 1,
                                  training=training,
                                  weights_fp16_on=conv_weights_fp16_on_l[11],
                                  bias_training=local_bn_training)
            return x

    def init_block(self, x):
        conv_bias = False
        local_training = False if self.fix_blocks >= 1 else self.training
        local_bn_training = self.bn_training if local_training else False
        local_conv_fp16 = False
        local_bn_fp16 = False
        x = my_conv(x,
                    7,
                    2,
                    3,
                    64,
                    conv_bias,
                    1,
                    fp16_on=local_conv_fp16,
                    training=local_training,
                    bias_training=local_bn_training,
                    debugPrefix='conv1')
        x = gcop.layers.batch_normalization(x,
                                            axis=1,
                                            trainable=local_bn_training,
                                            fp16_on=local_bn_fp16,
                                            name='bn1')
        x = gcop.nn.relu(x)

        x = gcop.layers.max_pooling2d(x,
                                      strides=2,
                                      pool_size=3,
                                      padding='same',
                                      data_format='channels_first')
        return x

    def layerX(self, x, index):
        local_training = False if index + 1 <= self.fix_blocks else self.training
        cast_flag, x, fp16_on = gcop.bF.deduce_half(x, cfg.TRAIN.LAYERXS_FP16[index])
        result = self.bottleneck_block(x,
                                       stride=self.layers_stride[index],
                                       c_out=self.layers_out[index],
                                       count=self.layers_count[index],
                                       debugPrefix='layer{}'.format(index + 1),
                                       training=local_training)
        if cast_flag:
            result = result.cast(cast_flag)
        return result

    def __forward__(self, x):
        #
        x = self.init_block(x)
        x = self.layerX(x, 0)
        x = self.layerX(x, 1)
        x = self.layerX(x, 2)

        return [x]

    def cls_reg_head(self, feature):
        if cfg.TRAIN.FC_FP16 is False:
            feature = feature.cast(gcop.float32)
        with gcop.variable_scope("cls_head"):
            cls_score = gcop.cF.fc(feature,
                                   num_units_out=self.classes_n,
                                   weights=self.normal_init(
                                       [feature.pureShape[1], self.classes_n], 0, 0.01, dtype=self.dtype),
                                   train=True,
                                   fp16_on=self.fc_fp16)

        with gcop.variable_scope("reg_head"):
            channels = 4 * self.classes_n if cfg.MODEL.RCNN.EXPAND_PREDICTED_BOXES else 4
            bbox_pred = gcop.cF.fc(feature,
                                   num_units_out=channels,
                                   weights=self.normal_init([feature.pureShape[1], channels], 0, 0.001,
                                                            dtype=self.dtype),
                                   train=True,
                                   fp16_on=self.fc_fp16)

        return cls_score, bbox_pred

    def __call__(self, x):
        return self.__forward__(x)
