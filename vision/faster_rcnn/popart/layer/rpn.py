# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""The RPN related classes.

The RPN class contain the cls head and reg head.
The Proposal class contain the box process code
example: apply box detal and nms.
Modified by Hu Di
"""

import numpy as np
from config import cfg
import ctypes
import os
from layer.base import BaseModel, smooth_l1_loss, get_valid_area_mask
import math
from IPU.ipu_tensor import gcop
from layer.base import DetectBase, nms


class Proposal(BaseModel, DetectBase):
    """this class used for post process the rpn result.
    example:
        crop to image size and nms.
    """

    def __init__(
        self,
        fp16_on=False,
        feat_stride=16,
        scales=[8, 16, 32],
        ratios=[0.5, 1, 2],
        input_size=[512, 512],
        training=True,
    ):
        """feat_stride: [N,BoxNum,5]"""

        super().__init__(fp16_on=fp16_on, training=training)
        # [1, 12960, 4]
        feat_width, feat_height = math.ceil(
            input_size[0] / feat_stride), math.ceil(input_size[1] /
                                                    feat_stride)
        self.boxes = self.process(
            self.generate_anchors(scales=np.array(scales),
                                  ratios=np.array(ratios)),
            feat_size=np.asarray([feat_height, feat_width]),
            feat_stride=feat_stride).astype(self.dtype)

        self.anchors = gcop.constant(self.boxes)

        self.num_anchors = len(scales)*len(ratios)
        self.batch_size = 1

    def bbox_transform_inv(self, boxes, deltas):
        """All boxes and deltas is a Tensor name on IPU.

        args:
            boxes: [1, 12960, 4]
            deltas: [1, 12960, 4]
        """
        #
        with gcop.variable_scope("bbox_transform_inv"):
            widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
            heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0

            ctr_x = boxes[:, :, 0] + 0.5 * widths
            ctr_y = boxes[:, :, 1] + 0.5 * heights

            dx = deltas[:, :, 0]
            dy = deltas[:, :, 1]
            dw = deltas[:, :, 2]
            dh = deltas[:, :, 3]

            pred_ctr_x = dx * widths + ctr_x
            pred_ctr_y = dy * heights + ctr_y
            pred_w = gcop.exp(dw) * widths
            pred_h = gcop.exp(dh) * heights

            x1 = gcop.expand_dims(pred_ctr_x - 0.5 * pred_w, -1)
            y1 = gcop.expand_dims(pred_ctr_y - 0.5 * pred_h, -1)
            x2 = gcop.expand_dims(0.5 * pred_w + pred_ctr_x, -1)
            y2 = gcop.expand_dims(0.5 * pred_h + pred_ctr_y, -1)

            pred_boxes = gcop.concat([x1, y1, x2, y2], 2)

        return pred_boxes

    def clip_boxes(self, boxes, im_info):
        with gcop.variable_scope("clip_boxes"):
            x1 = gcop.clip_by_value(boxes[:, :, 0],
                                    clip_value_min=0,
                                    clip_value_max=im_info[1] -
                                    1).unsqueeze(-1)
            y1 = gcop.clip_by_value(boxes[:, :, 1],
                                    clip_value_min=0,
                                    clip_value_max=im_info[0] -
                                    1).unsqueeze(-1)
            x2 = gcop.clip_by_value(boxes[:, :, 2],
                                    clip_value_min=0,
                                    clip_value_max=im_info[1] -
                                    1).unsqueeze(-1)
            y2 = gcop.clip_by_value(boxes[:, :, 3],
                                    clip_value_min=0,
                                    clip_value_max=im_info[0] -
                                    1).unsqueeze(-1)

            boxes = gcop.concat([x1, y1, x2, y2], 2)
        return boxes

    def __forward__(self, x, training):
        """Algorithm:

        1:for each (H, W) location i
            generate A anchor boxes centered on cell i.
        2:apply predicted bbox deltas at cell i to each of the A anchors
            clip predicted boxes to image.
        3:remove predicted boxes with either height or width < threshold
            sort all (proposal, score) pairs by score from highest to lowest.
        4:take top pre_nms_topN proposals before NMS.
        5:apply NMS with threshold 0.7 to remaining proposals
            take after_nms_topN proposals after NMS.
        return the top proposals (-> RoIs top, scores top)

        args:
            x[0]: rpn_cls_prob [1, 18, 30, 48]
            x[1]: rpn_bbox_pred [1, 36, 30, 48]
        """
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        with gcop.variable_scope("Proposal"):
            x[0] = x[0].detach()
            x[1] = x[1].detach()
            scores = x[0][:, self.num_anchors:, :, :]
            bbox_deltas = x[1]
            im_info = x[2]

            bbox_deltas = gcop.reshape(
                bbox_deltas, [self.batch_size, -1, 4])

            scores = gcop.transpose(scores, perm=[0, 2, 3,
                                                  1])
            B, C, H, W = scores.shape.as_list()
            scores = gcop.reshape(scores,
                                  [B, C * H * W])
            proposals = self.bbox_transform_inv(
                self.anchors,
                bbox_deltas)
            clipped_proposals = self.clip_boxes(
                proposals, im_info)
            valid_area_boxes = get_valid_area_mask(clipped_proposals)
            self.add_output('clipped_valid_area_boxes',
                            gcop.reduce_sum(valid_area_boxes))

            if cfg.TRAIN.RPN_PRE_NMS_TOP_N > 0:
                rpn_pre_nms_top_n = min(
                    cfg.TRAIN.RPN_PRE_NMS_TOP_N, scores.squeeze(0).pureShape[0])
                sorted_scores, order = gcop.nn.top_k(
                    scores.squeeze(0), k=rpn_pre_nms_top_n)
                sorted_clipped_proposals = gcop.gather(
                    clipped_proposals, order, axis=1)
            else:
                sorted_scores = scores.squeeze(0)
                sorted_clipped_proposals = clipped_proposals

            output_boxes, output_keeps, _ = nms(
                sorted_scores.unsqueeze(0),
                sorted_clipped_proposals,
                numDetections=cfg.TRAIN.RPN_POST_NMS_TOP_N
                if training else cfg.TEST.RPN_POST_NMS_TOP_N)

            valid_area_output_boxes = get_valid_area_mask(output_boxes)
            self.add_output('valid_area_output_boxes',
                            gcop.reduce_sum(valid_area_output_boxes))
        return output_boxes, output_keeps

    def __call__(self, x, training):
        return self.__forward__(x, training)


class RPN(BaseModel):
    def __init__(
        self,
        classes,
        fp16_on=False,
        training=True,
        rpn_channel=512,
        input_size=[600, 600],
    ):
        super().__init__(fp16_on=fp16_on, training=training)
        self.proposal = Proposal(fp16_on=False,
                                 scales=cfg.ANCHOR_SCALES,
                                 ratios=cfg.ANCHOR_RATIOS,
                                 input_size=input_size,
                                 feat_stride=cfg.FEAT_STRIDE,
                                 training=training)
        self.classes = classes
        self.classes_count = len(classes)
        self.anchor_scales = np.array(cfg.ANCHOR_SCALES)
        self.anchor_ratios = np.array(cfg.ANCHOR_RATIOS).astype(self.dtype)
        self.nc_score_out = len(self.anchor_scales) * \
            len(self.anchor_ratios) * 2
        self.rpn_channel = rpn_channel

    def forward(self, x, im_info=None, rpn_data=None, stage_configs='0'):
        if cfg.MODEL.RPN_CONV_FP16_ON:
            x = x.cast(gcop.float16)
        else:
            x = x.cast(gcop.float32)

        with gcop.variable_scope("rpn"):
            x = gcop.cF.conv2d(x,
                               self.rpn_channel,
                               ksize=3,
                               train=True,
                               strides=[1, 1],
                               padding_mode='same',
                               fp16_on=None,
                               weights_fp16_on=cfg.MODEL.RPN_CONV_FP16_ON,
                               filters_data=self.normal_init(
                                   [self.rpn_channel, x.pureShape[1], 3, 3], 0, 0.01, dtype=self.dtype),
                               debugContext='conv')
            x = gcop.nn.relu(x)
            with gcop.variable_scope("rpn_cls"):
                rpn_cls_score = gcop.cF.conv2d(x,
                                               self.nc_score_out,
                                               ksize=1,
                                               train=True,
                                               strides=[1, 1],
                                               padding_mode='same',
                                               fp16_on=None,
                                               filters_data=self.normal_init([self.nc_score_out, x.pureShape[1], 1, 1], 0, 0.01, dtype=self.dtype))

                B, C, H, W = rpn_cls_score.shape.as_list()
                target_shape = [B, 2, -1]
                rpn_cls_score_reshape = gcop.reshape(
                    rpn_cls_score, target_shape)
                rpn_cls_prob_premute = gcop.transpose(rpn_cls_score_reshape,
                                                      perm=[0, 2,
                                                            1])
                rpn_cls_prob_premute_reshape = gcop.reshape(
                    rpn_cls_prob_premute, [-1, 2])
                rpn_cls_prob_premute_reshape = rpn_cls_prob_premute_reshape.cast(
                    gcop.float32)

                logits = gcop.nn.softmax(
                    rpn_cls_prob_premute_reshape)

            # get rpn offsets to the anchor boxes
            with gcop.variable_scope("rpn_box"):
                rpn_bbox_pred = gcop.cF.conv2d(x,
                                               self.nc_score_out * 2,
                                               ksize=1,
                                               train=True,
                                               strides=[1, 1],
                                               padding_mode='same',
                                               fp16_on=None,
                                               filters_data=self.normal_init([self.nc_score_out*2, x.pureShape[1], 1, 1], 0, 0.01, dtype=self.dtype))
                rpn_bbox_pred = gcop.transpose(rpn_bbox_pred,
                                               [0, 2, 3, 1])

            if cfg.MODEL.RPN_CONV_FP16_ON:
                rpn_bbox_pred = rpn_bbox_pred.cast(gcop.float32)
                logits = logits.cast(gcop.float32)
                rpn_cls_prob_premute_reshape = rpn_cls_prob_premute_reshape.cast(
                    gcop.float32)

        with gcop.device(stage_configs):
            if self.training:
                _rpn_label, rpn_keep, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data
                rpn_keep = rpn_keep.squeeze(0)

                rpn_scores = gcop.gather(
                    rpn_cls_prob_premute_reshape,
                    rpn_keep.cast(gcop.int32),
                )

                rpn_label = gcop.gather(
                    gcop.reshape(_rpn_label, [-1]),
                    rpn_keep.cast(gcop.int32),
                ).cast(gcop.int32)

                rpn_scores, rpn_bbox_pred = [
                    ele.cast(gcop.float32) for ele in [rpn_scores, rpn_bbox_pred]
                ]

                self.rpn_loss_cls = gcop.nn.sparse_softmax_cross_entropy_with_logits(  # noqa
                    labels=rpn_label, logits=rpn_scores, name="rpn_loss_cls")

                self.rpn_loss_box = smooth_l1_loss(
                    rpn_bbox_pred,
                    rpn_bbox_targets,
                    rpn_bbox_inside_weights,
                    rpn_bbox_outside_weights,
                    sigma=3,
                    reduceDim=[0, 1, 2, 3],
                    debugPrefix='rpn_loss_box')

            else:
                self.rpn_loss_cls, self.rpn_loss_box = 0, 0

            logits_transpose = gcop.transpose(logits, perm=[1, 0])
            rpn_cls_prob = gcop.reshape(logits_transpose,
                                        [B, C, H, W])

            fixed_length_roi, roi_keeps = self.proposal(
                [rpn_cls_prob, rpn_bbox_pred, im_info],
                self.training)

        return fixed_length_roi, roi_keeps, self.rpn_loss_cls, self.rpn_loss_box
