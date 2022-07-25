#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

from functools import partial

import numpy as np

import core.backbone as backbone
import core.common as common
import core.utils as utils
import ipu_utils
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ops import array_ops, math_ops


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, trainable, opts):

        self.trainable = trainable
        self.classes = utils.read_class_names(opts["yolo"]["classes"])
        self.num_class = len(self.classes)
        self.strides = np.array(opts["yolo"]["strides"])
        self.anchors = utils.get_anchors(opts["yolo"]["anchors"])
        self.anchor_per_scale = opts["yolo"]["anchor_per_scale"]
        self.iou_loss_thresh = opts["yolo"]["iou_loss_thresh"]
        self.upsample_method = opts["yolo"]["upsample_method"]
        self.precision = tf.float16 if opts["yolo"]["precision"] == "fp16" else tf.float32
        self.opts = opts
        self.darknet_gn = opts["yolo"]["darknet_gn"]
        self.upsample_gn = opts["yolo"]["upsample_gn"]
        self.use_centering = opts["yolo"]["use_centering"]

    def decode_boxes(self, conv_lbbox, conv_mbbox, conv_sbbox):
        """Decode boxes from anchor coordinates to grid coordinates for all scales.
        Args:
            conv_lbbox: Conv output for large bboxes
            conv_mbbox: Conv output for middle bboxes
            conv_sbbox: Conv output for small bboxes
            pad_s: Padding mutliplier for small boxes. it's value will be one of {1.0, 0.0}.
                one for image points, 0 for padded points, to avoid padded values affect param gradients
            pad_m: Padding multiplier for middle boxes. content same as pad_s
            pad_l: Padding multiplier for large boxes. content same as pad_s
        """
        with tf.variable_scope("pred_sbbox"):
            pred_sbbox = self.decode(
                conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope("pred_mbbox"):
            pred_mbbox = self.decode(
                conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope("pred_lbbox"):
            pred_lbbox = self.decode(
                conv_lbbox, self.anchors[2], self.strides[2])

        return {"pred_sbbox": pred_sbbox, "pred_mbbox": pred_mbbox, "pred_lbbox": pred_lbbox}

    def build_backbone(self):
        return backbone.darknet53(self.trainable, self.darknet_gn, self.precision)

    def build_upsample(self):
        funcs = []

        wrapper = ipu_utils.convolutional(
            (1, 1, 1024,  512), self.trainable, self.upsample_gn, "conv52", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (3, 3,  512, 1024), self.trainable, self.upsample_gn, "conv53", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (1, 1, 1024,  512), self.trainable, self.upsample_gn, "conv54", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (3, 3,  512, 1024), self.trainable, self.upsample_gn, "conv55", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (1, 1, 1024,  512), self.trainable, self.upsample_gn, "conv56", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)

        wrapper = ipu_utils.convolutional(
            (3, 3, 512, 1024), self.trainable, self.upsample_gn, name="conv_lobj_branch", weight_centering=self.use_centering, precision=self.precision)
        wrapper = ipu_utils.branch(wrapper, "conv_lobj_branch")
        funcs.append(wrapper)

        def head_1_wrapper(conv_lobj_branch):
            conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                              trainable=self.trainable, use_gn=self.upsample_gn, name="conv_lbbox", activate=False, norm=False, weight_centering=self.use_centering, precision=self.precision)
            return {"conv_lbbox": conv_lbbox}
        funcs.append(head_1_wrapper)

        wrapper = ipu_utils.convolutional(
            (1, 1,  512,  256), self.trainable, self.upsample_gn, "conv57", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)

        def upsample_1_wrapper(route_2, input_data):
            input_data = common.upsample(input_data, name="upsample0",
                                         method=self.upsample_method, precision=self.precision)
            with tf.variable_scope("route_1"):
                input_data = tf.concat([input_data, route_2], axis=-1)
                return {"input_data": input_data}
        funcs.append(upsample_1_wrapper)

        wrapper = ipu_utils.convolutional(
            (1, 1, 768, 256), self.trainable, self.upsample_gn, "conv58", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (3, 3, 256, 512), self.trainable, self.upsample_gn, "conv59", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (1, 1, 512, 256), self.trainable, self.upsample_gn, "conv60", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (3, 3, 256, 512), self.trainable, self.upsample_gn, "conv61", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (1, 1, 512, 256), self.trainable, self.upsample_gn, "conv62", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (3, 3, 256, 512),  self.trainable, self.upsample_gn, name="conv_mobj_branch", weight_centering=self.use_centering, precision=self.precision)
        wrapper = ipu_utils.branch(wrapper, "conv_mobj_branch")
        funcs.append(wrapper)

        def head_2_wrapper(conv_mobj_branch):
            conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                              trainable=self.trainable, use_gn=self.upsample_gn, name="conv_mbbox", activate=False, norm=False, weight_centering=self.use_centering, precision=self.precision)
            return {"conv_mbbox": conv_mbbox}
        funcs.append(head_2_wrapper)

        wrapper = ipu_utils.convolutional(
            (1, 1, 256, 128), self.trainable, self.upsample_gn, "conv63", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)

        def upsample_2_wrapper(route_1, input_data):
            input_data = common.upsample(input_data, name="upsample1",
                                         method=self.upsample_method, precision=self.precision)
            with tf.variable_scope("route_2"):
                input_data = tf.concat([input_data, route_1], axis=-1)
            return {"input_data": input_data}
        funcs.append(upsample_2_wrapper)

        wrapper = ipu_utils.convolutional(
            (1, 1, 384, 128), self.trainable, self.upsample_gn, "conv64", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (3, 3, 128, 256), self.trainable, self.upsample_gn, "conv65", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (1, 1, 256, 128), self.trainable, self.upsample_gn, "conv66", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (3, 3, 128, 256), self.trainable, self.upsample_gn, "conv67", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)
        wrapper = ipu_utils.convolutional(
            (1, 1, 256, 128), self.trainable, self.upsample_gn, "conv68", weight_centering=self.use_centering, precision=self.precision)
        funcs.append(wrapper)

        wrapper = ipu_utils.convolutional(
            (3, 3, 128, 256), self.trainable, self.upsample_gn, name="conv_sobj_branch", weight_centering=self.use_centering, precision=self.precision)
        wrapper = ipu_utils.branch(wrapper, "conv_sobj_branch")
        funcs.append(wrapper)

        def head_3_wrapper(conv_sobj_branch):
            conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                              trainable=self.trainable, use_gn=self.upsample_gn, name="conv_sbbox", activate=False, norm=False, weight_centering=self.use_centering, precision=self.precision)
            return {"conv_sbbox": conv_sbbox}
        funcs.append(head_3_wrapper)
        return funcs

    def decode(self, conv_output, anchors, stride):
        """Decode boxes from anchor coordinate to grid coordinate
        Args:
            conv_output: output of convolutional predict head
        Returns:
            tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
        """

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(
            conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)
                    [:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)
                    [tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat(
            [x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [
                          batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, self.precision)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):
        """GIoU function:https://arxiv.org/pdf/1902.09630.pdf
        Args:
            boxes1:
                tensor of boxes, elements of last dimension is x,y,w,h of the boxes
            boxes2:
                another tensor of boxes
        """
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # NOTE: why is there a minimum?
        boxes1 = tf.concat([self.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            self.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([self.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            self.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
            (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
            (boxes2[..., 3] - boxes2[..., 1])

        left_up = self.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = self.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = self.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        # union_area maybe zero for giou
        iou = inter_area / tf.where(union_area <= tf.constant(0.1, shape=union_area.shape, dtype=union_area.dtype),
                                    tf.constant(1.0, shape=union_area.shape, dtype=union_area.dtype), union_area)

        enclose_left_up = self.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = self.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = self.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def maximum(self, tensor1, tensor2):
        """out lined maximum function
        Note:
            we use this funciton for reduce memory spikes on loss function
        """

        @ipu.outlined_function
        def max_func(tensor1, tensor2):
            return tf.maximum(tensor1, tensor2)
        if isinstance(tensor2, float):
            tensor2 = np.float32(tensor2)
        return max_func(tensor1, tensor2)

    def minimum(self, tensor1, tensor2):
        """out lined maximum function
        Note:
            we use this funciton for reduce memory spikes on loss function
        """
        @ipu.outlined_function
        def min_func(tensor1, tensor2):
            return tf.minimum(tensor1, tensor2)
        if isinstance(tensor2, float):
            tensor2 = np.float32(tensor2)
        return min_func(tensor1, tensor2)

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = self.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = self.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = self.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area
        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        """Loss function all scales
        Args:
            conv:
        """

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, self.precision)

        bbox_loss_scale = 2.0 - 1.0 * \
            label_xywh[:, :, :, :, 2:3] * \
            label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                            bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou <
                                                     self.iou_loss_thresh, self.precision)

        conf_focal = self.focal(respond_bbox, pred_conf)

        # objectness confidence loss
        conf_loss = conf_focal * (
            respond_bbox *
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=respond_bbox, logits=conv_raw_conf) +
            respond_bgd *
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=respond_bbox, logits=conv_raw_conf)
        )

        # class probability loss
        prob_loss = respond_bbox * \
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self,
                     conv_sbbox,
                     conv_mbbox,
                     conv_lbbox,
                     pred_sbbox,
                     pred_mbbox,
                     pred_lbbox,
                     label_sbbox,
                     label_mbbox,
                     label_lbbox,
                     true_sbbox,
                     true_mbbox,
                     true_lbbox):
        """loss function for all scales
        Args:
            conv_sbbox: Small bboxes with  anchor relative coordinates.
            conv_mbbox: Middle bboxes with  anchor relative coordinates.
            conv_lbbox: Large bboxes with  anchor relative coordinates.
            pred_sbbox: Small bboxes with grid relative coordinates.
                tensor shape (batch_size, grid_size, grid_size, anchors_per_point, classes+objectness+bbox_coordinates)
            pred_mbbox: Middle bboxes with grid relative coordinates.
            pred_lbbox: Large bboxes with grid relative coordinates.
            label_sbbox: Small boxes label for every anchor. shape same as pred_sbbox
            label_mbbox: Middle boxes label for every anchor.
            label_lbbox: Large boxes label for every anchor.
            true_sbbox: Small boxes in shape (batch_size, max_box_per_scale, coordinates).
            true_mbbox: Middle boxes in shape (batch_size, max_box_per_scale, coordinates).
            true_lbbox: Large boxes in shape (batch_size, max_box_per_scale, coordinates).
        """
        # we use fp32 for loss computation
        # because we found that fp16 may produce nan
        # map values to fp32
        conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox = map(partial(
            tf.cast, dtype=tf.float32), [conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox])
        self.precision = tf.float32
        with tf.name_scope("smaller_box_loss"):
            loss_sbbox = self.loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope("medium_box_loss"):
            loss_mbbox = self.loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope("bigger_box_loss"):
            loss_lbbox = self.loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope("giou_loss"):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope("conf_loss"):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope("prob_loss"):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        # set precision back to default value
        self.precision = tf.float16
        return {"giou_loss": tf.cast(giou_loss, dtype=self.precision), "conf_loss": tf.cast(conf_loss, dtype=self.precision), "prob_loss": tf.cast(prob_loss, dtype=self.precision)}
