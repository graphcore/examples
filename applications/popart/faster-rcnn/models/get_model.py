# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from .faster_rcnn import FasterRcnn


def make_model(model_name='FasterRcnn', **params):
    if model_name == 'FasterRcnn':
        return get_faster_rcnn(**params)
    else:
        raise NotImplementedError


def get_faster_rcnn(input_im_shape=[1, 3, 512, 512],
                    input_box_num=20,
                    fp16_on=False,
                    classes=[1] * 21,
                    training=True):
    faster_rcnn = FasterRcnn(
        input_im_shape=input_im_shape,
        fp16_on=fp16_on,
        input_box_num=input_box_num,
        training=training,
        classes=classes,
    )
    return faster_rcnn
