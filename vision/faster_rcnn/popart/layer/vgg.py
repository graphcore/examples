# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from layer.base import BaseModel


class VGG16(BaseModel):
    """The VGG16 base backbone on IPU with Popart
    Paper reference:https://arxiv.org/abs/1409.1556
    More info is at:http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    NOTICE:
        diffrent from origin VGG the first conv layer is 3*3
        and remove the last pooling that was used in Origin VGG
    """

    def __init__(self,
                 classes,
                 initializer={},
                 builder=None,
                 fp16_on=False,
                 available_memory_proportion=None):
        """
        args:
            classes:            How many categories are to be classified
            initializer:        init map that obtained from
                                utils.get_initializer_from_pth
            dtype:              model weights type .default is float32
            available_memory_proportion:
        """
        super().__init__(builder, initializer, fp16_on,
                         available_memory_proportion)
        self.classes = classes
        self.classes_n = len(self.classes)

    def __vgg_classifier__(self, x):
        """The VGG classifier header.
        remove dropout for inference
        remove the last fc layer for
        faster rcnn.
        """
        with self.namescope("top_fc"):
            with self.namescope("fc1"):
                x = self.fc(x, 2048)
                x = self.relu(x)
            with self.namescope("fc2"):
                x = self.fc(x, 2048)
                x = self.relu(x)
        return x

    def conv_relu(self,
                  x,
                  ksize,
                  stride,
                  pads,
                  c_out,
                  debugPrefix,
                  bias=True,
                  group=1,
                  relu6=False):
        """conv with relu for sample."""

        with self.namescope(debugPrefix):
            x = self.conv(
                x,
                ksize=ksize,
                stride=stride,
                pads=pads,
                c_out=c_out,
                debugPrefix=debugPrefix,
                bias=bias,
                group=group,
            )

            x = self.relu(x, relu6)

        return x

    def __forward__(self, x):
        """init sub class graph."""
        with self.namescope("vgg"):
            x = self.conv_relu(x, 3, 1, 1, 64, "conv1")
            x = self.conv_relu(x, 3, 1, 1, 64, "conv2")
            x = self.maxPooling(x, 2, 2, 0)

            x = self.conv_relu(x, 3, 1, 1, 128, "conv3")
            x = self.conv_relu(x, 3, 1, 1, 128, "conv4")
            x = self.maxPooling(x, 2, 2, 0)

            x = self.conv_relu(x, 3, 1, 1, 256, "conv5")
            x = self.conv_relu(x, 3, 1, 1, 256, "conv6")
            x = self.conv_relu(x, 3, 1, 1, 256, "conv7")
            x = self.maxPooling(x, 2, 2, 0)

            x = self.conv_relu(x, 3, 1, 1, 512, "conv8")
            x = self.conv_relu(x, 3, 1, 1, 512, "conv9")
            x = self.conv_relu(x, 3, 1, 1, 512, "conv10")
            x = self.maxPooling(x, 2, 2, 0)

            x = self.conv_relu(x, 3, 1, 1, 512, "conv11")
            x = self.conv_relu(x, 3, 1, 1, 512, "conv12")
            x = self.conv_relu(x, 3, 1, 1, 512, "conv13")
            return [x]

    def head_to_tail(self, pool5):
        """pool5: [300, 512, 7, 7]."""
        pool5_shape = self.getTensorShape(pool5)
        pool5_flat = self.reshape(pool5, [pool5_shape[0], -1])
        fc7 = self.__vgg_classifier__(pool5_flat)
        return fc7

    def cls_reg_head(self, feature):
        with self.namescope("cls_head"):
            cls_head = self.fc(feature, self.classes_n)
            cls_prob = self.builder.aiOnnx.softmax([cls_head], axis=1)

        with self.namescope("reg_head"):
            bbox_pred = self.fc(feature, self.classes_n * 4)

        return cls_prob, bbox_pred

    def __call__(self, x):
        return self.__forward__(x)
