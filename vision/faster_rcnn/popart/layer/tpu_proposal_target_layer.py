# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
# This file has been modified by Graphcore Ltd.
from logging import debug
import numpy as np
from layer.base import BaseModel, bbox_overlaps_torch, bbox_overlaps_tf
from config import cfg
from . import balanced_positive_negative_sampler
from IPU.ipu_tensor import gcop
'''
reference code: https://github.com/tensorflow/tpu/blob/3679ca6b979349dde6da7156be2528428b000c7c/models/official/mask_rcnn/training_ops.py
'''


def bbox_transform(ex_rois, gt_rois):
    assert ex_rois.shape.ndims == 3
    assert gt_rois.shape.ndims == 3
    ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
    ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
    ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
    gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
    gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = gcop.log(gt_widths / ex_widths)
    targets_dh = gcop.log(gt_heights / ex_heights)

    targets = gcop.stack((targets_dx, targets_dy, targets_dw, targets_dh), 2)
    return targets


class ProposalTargetLayer(BaseModel):
    def __init__(self,
                 roi_thrd=0.5,
                 batch_size_per_im=256,
                 num_classes=21,
                 fp16_on=False,
                 positive_fraction=0.5,
                 training=True):

        super().__init__(fp16_on=fp16_on, training=training)
        self.roi_thrd = gcop.constant(np.asarray(roi_thrd).astype(np.float32))
        self.zero_constant = gcop.constant(np.asarray(0).astype(np.float32))
        self.batch_size_per_im = batch_size_per_im
        self._positive_fraction = positive_fraction
        self.num_pos = int(batch_size_per_im * positive_fraction)
        self.num_neg = batch_size_per_im - self.num_pos
        self.num_classes = num_classes

    def __call__(self, fixed_length_roi, roi_keeps, gt_boxes):
        return self.forward(fixed_length_roi, roi_keeps, gt_boxes)

    def labels2mask(self, labels):
        mask = gcop.greater(labels, gcop.zeros_like(labels))
        return mask

    def expand_target_boxes(self, encoded_rois_target, class_labels,
                            num_classes):
        # encoded_rois_target: 1x256x4
        # class_labels: 1x256
        # num_classes: 21(int)
        # return 1x256x84(21x4)
        encoded_rois_target_unsqueeze = gcop.expand_dims(
            encoded_rois_target, 2)  # 1x256x1x4
        encoded_rois_target_tile = gcop.tile(
            encoded_rois_target_unsqueeze,
            [1, 1, num_classes, 1])  # 1x256x21x4
        class_labels_int = class_labels.cast(gcop.int32)
        class_labels_ont_hot = gcop.one_hot(class_labels_int,
                                            num_classes)  # 1x256x21
        class_labels_unsqueeze = class_labels_ont_hot.unsqueeze(-1).cast(
            gcop.float32)  # 1x256x21x1
        encoded_rois_target_masked = class_labels_unsqueeze * \
            encoded_rois_target_tile  # 1x256x21x4
        batch, num_boxes, _, _ = encoded_rois_target_masked.shape.as_list(
        )
        encoded_rois_target_expand = encoded_rois_target_masked.reshape(
            [batch, num_boxes, -1])
        return encoded_rois_target_expand

    def forward(self,
                fixed_length_roi,
                roi_keeps,
                gt_proposals):
        #
        fixed_length_roi = gcop.cast(fixed_length_roi.detach(), gcop.float32)
        gt_proposals = gcop.cast(gt_proposals.detach(), gcop.float32)

        gt_boxes = gt_proposals[:, :, :4]
        gt_labels = gt_proposals[:, :, 4:]

        sample_box_targets, class_targets, rois, sample_proposal_to_label_map, encoded_rois_target = self.rois_sampler(
            fixed_length_roi,
            gt_boxes,
            gt_labels,
            batch_size_per_im=self.batch_size_per_im,
            fg_fraction=self._positive_fraction,
            fg_thresh=self.roi_thrd,
            bg_thresh_hi=self.roi_thrd,
            bg_thresh_lo=cfg.TRAIN.BG_THRESH_LO,
        )
        positives_mask = self.labels2mask(class_targets)

        # set bbox_inside_weights, bbox_outside_weights
        _batch, _num_boxes = positives_mask.shape.as_list()
        _weights = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
        _weights = np.asarray(_weights, dtype=np.float32)[np.newaxis,
                                                          np.newaxis, :]
        _weights = np.tile(_weights, (_batch, _num_boxes, 1))
        bbox_inside_weights = gcop.constant(
            _weights) * positives_mask.unsqueeze(-1).cast(gcop.float32)

        if cfg.MODEL.RCNN.EXPAND_PREDICTED_BOXES:
            encoded_rois_target = self.expand_target_boxes(
                encoded_rois_target, class_targets, self.num_classes)
            bbox_inside_weights = self.expand_target_boxes(
                bbox_inside_weights, class_targets, self.num_classes)
            sample_box_targets = self.expand_target_boxes(
                sample_box_targets, class_targets, self.num_classes)

        return_list = [
            rois, class_targets, sample_box_targets,
            sample_proposal_to_label_map, positives_mask, encoded_rois_target,
            bbox_inside_weights
        ]

        return return_list

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        targets = bbox_transform(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            targets = targets - gcop.constant(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32))
            targets = targets / gcop.constant(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32))
        return targets

    def get_valid_area_flags(self, boxes):
        # input boxes: 1,n,4
        # output: mask: 1,n
        ws = boxes[:, :, 2] - boxes[:, :, 0]  # 1,n
        hs = boxes[:, :, 3] - boxes[:, :, 1]  # 1,n
        areas = ws * hs  # 1,n
        valid_flags = gcop.greater(
            areas, gcop.constant(np.asarray(0.0, dtype=np.float32)))
        return valid_flags

    def rois_sampler(self,
                     boxes,
                     gt_boxes,
                     gt_labels,
                     batch_size_per_im=512,
                     fg_fraction=0.25,
                     fg_thresh=0.5,
                     bg_thresh_hi=0.5,
                     bg_thresh_lo=0.0):
        """Assigns the proposals with ground truth labels and performs subsmpling.
        Given proposal `boxes`, `gt_boxes`, and `gt_labels`, the function uses the
        following algorithm to generate the final `batch_size_per_im` RoIs.
        1. Calculates the IoU between each proposal box and each gt_boxes.
        2. Assigns each proposal box with a ground truth class and box label by
            choosing the largest overlap.
        3. Samples `batch_size_per_im` boxes from all proposal boxes, and returns
            box_targets, class_targets, and RoIs.
        The reference implementations of #1 and #2 are here: https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py  # pylint: disable=line-too-long
        The reference implementation of #3 is here: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py.  # pylint: disable=line-too-long
        Args:
            boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
            proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
            last dimension is the pixel coordinates of scaled images in
            [ymin, xmin, ymax, xmax] form.
            gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
            tensor might have paddings with a value of -1. The coordinates of gt_boxes
            are in the pixel coordinates of the scaled image.
            gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
            tensor might have paddings with a value of -1.
            batch_size_per_im: an integer represents RoI minibatch size per image.
            fg_fraction: a float represents the target fraction of RoI minibatch that
            is labeled foreground (i.e., class > 0).
            fg_thresh: a float represents the overlap threshold for an RoI to be
            considered foreground (if >= fg_thresh).
            bg_thresh_hi: a float represents the overlap threshold for an RoI to be
            considered background (class = 0 if overlap in [LO, HI)).
            bg_thresh_lo: a float represents the overlap threshold for an RoI to be
            considered background (class = 0 if overlap in [LO, HI)).
        Returns:
            box_targets: a tensor with a shape of [batch_size, K, 4]. The tensor
            contains the ground truth pixel coordinates of the scaled images for each
            roi. K is the number of sample RoIs (e.g., batch_size_per_im).
            class_targets: an integer tensor with a shape of [batch_size, K]. The tensor
            contains the ground truth class for each roi. Note, 0 for background, 1 to N
            represent N obj classes.
            rois: a tensor with a shape of [batch_size, K, 4], representing the
            coordinates of the selected RoI.
            proposal_to_label_map: a tensor with a shape of [batch_size, K]. This tensor
            keeps the mapping between proposal to labels. proposal_to_label_map[i]
            means the index of the ground truth instance for the i-th proposal. For example,
            -1 for no obj, 0 for first instance, 1 for second instance.
        """
        with gcop.variable_scope('ProposalTargetLayer'):
            batch_size = boxes.shape.as_list()[0]

            # The reference implementation intentionally includes ground truth boxes in
            # the proposals. see https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py#L359.  # pylint: disable=line-too-long
            if cfg.TRAIN.ADD_GT_BOX_IN_SAMPLER:
                boxes = gcop.concat([boxes, gt_boxes], axis=1)
            else:
                pass
            boxes_keep_arr = self.get_valid_area_flags(boxes)
            gt_boxes_keep_arr = self.get_valid_area_flags(gt_boxes)

            iou = bbox_overlaps_torch(boxes[0], gt_boxes[0])
            iou = iou.unsqueeze(0)
            iou_keep_arr = (boxes_keep_arr.cast(
                gcop.float32).unsqueeze(-1)) * (gt_boxes_keep_arr.cast(
                    gcop.float32).unsqueeze(1))
            iou = iou * iou_keep_arr

            (pre_sample_box_targets, pre_sample_class_targets, max_overlap,
             proposal_to_label_map) = self._add_class_assignments(
                 iou, gt_boxes, gt_labels)

            # Generates a random sample of RoIs comprising foreground and background
            # examples. reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py#L132  # pylint: disable=line-too-long
            positives = gcop.math.greater_equal(
                max_overlap.cast(gcop.float32),
                (fg_thresh * gcop.ones_like(max_overlap)).cast(gcop.float32))
            negatives = gcop.math.logical_and(
                gcop.math.greater_equal(
                    max_overlap, bg_thresh_lo * gcop.ones_like(max_overlap)),
                gcop.less(max_overlap,
                          bg_thresh_hi * gcop.ones_like(max_overlap)))
            pre_sample_class_targets = gcop.where(
                negatives, gcop.zeros_like(pre_sample_class_targets),
                pre_sample_class_targets)
            proposal_to_label_map = gcop.where(
                negatives,
                gcop.ones_like(proposal_to_label_map).cast(gcop.int32) * -1,
                proposal_to_label_map.cast(gcop.int32),
            )  # -1 for no instance in current proposal,
            # 0 for first instance(not class, there might be one class but 888 instances) of input targets

            # Handles ground truth paddings.
            ignore_mask = gcop.less(gcop.reduce_min(iou, axis=2),
                                    gcop.zeros_like(max_overlap))
            # indicator includes both positive and negative labels.
            # labels includes only positives labels.
            # positives = indicator & labels.
            # negatives = indicator & !labels.
            # ignore = !indicator.
            positive_flags = positives
            negative_flags = negatives
            pos_or_neg = gcop.math.logical_or(positives, negatives)
            indicator = gcop.math.logical_and(
                pos_or_neg, gcop.math.logical_not(ignore_mask))

            all_samples = []
            sampler = (balanced_positive_negative_sampler.
                       BalancedPositiveNegativeSampler(
                           fp16_on=self.fp16_on,
                           training=True,
                           positive_fraction=fg_fraction))
            # Batch-unroll the sub-sampling process.
            for i in range(batch_size):
                samples = sampler.subsample(indicator[i], batch_size_per_im,
                                            positive_flags[i],
                                            negative_flags[i],
                                            boxes_keep_arr[i])
                all_samples.append(samples)
            all_samples = gcop.stack(all_samples, axis=0)
            # A workaround to get the indices from the boolean tensors.
            _, samples_indices = gcop.nn.top_k(all_samples.cast(gcop.int32),
                                               k=batch_size_per_im,
                                               sorted=True)
            # Contructs indices for gather.
            samples_indices = gcop.reshape(
                samples_indices.cast(gcop.int32) + gcop.expand_dims(
                    gcop.range(batch_size, dtype=gcop.int32).cast(gcop.int32) *
                    boxes.shape.as_list()[1], 1).cast(gcop.int32),
                [-1]).cast(gcop.int32)

            rois = gcop.reshape(
                gcop.gather(gcop.reshape(boxes, [-1, 4]), samples_indices),
                [batch_size, -1, 4])

            class_targets = gcop.reshape(
                gcop.gather(gcop.reshape(pre_sample_class_targets, [-1, 1]),
                            samples_indices), [batch_size, -1])
            sample_box_targets = gcop.reshape(
                gcop.gather(gcop.reshape(pre_sample_box_targets, [-1, 4]),
                            samples_indices), [batch_size, -1, 4])
            sample_proposal_to_label_map = gcop.reshape(
                gcop.gather(gcop.reshape(proposal_to_label_map, [-1, 1]),
                            samples_indices), [batch_size, -1])

            encoded_boxes_result = self._compute_targets_pytorch(
                rois, sample_box_targets)

        return sample_box_targets, class_targets, rois, sample_proposal_to_label_map, encoded_boxes_result

    def _add_class_assignments(self, iou, gt_boxes, gt_labels):
        """Computes object category assignment for each box.
        Args:
            iou: a tensor for the iou matrix with a shape of
            [batch_size, K, MAX_NUM_INSTANCES]. K is the number of post-nms RoIs
            (i.e., rpn_post_nms_topn).
            gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4].
            This tensor might have paddings with negative values. The coordinates
            of gt_boxes are in the pixel coordinates of the scaled image scale.
            gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
            tensor might have paddings with a value of -1.
        Returns:
            max_boxes: a tensor with a shape of [batch_size, K, 4], representing
            the ground truth coordinates of each roi.
            max_classes: a int32 tensor with a shape of [batch_size, K], representing
            the ground truth class of each roi.
            max_overlap: a tensor with a shape of [batch_size, K], representing
            the maximum overlap of each roi.
            argmax_iou: a tensor with a shape of [batch_size, K], representing the iou
            argmax.
        """
        with gcop.variable_scope('add_class_assignments'):
            batch_size, _, _ = iou.shape.as_list()
            argmax_iou = gcop.argmax(
                iou,
                axis=2)
            local_interval = gcop.constant(
                np.array(gt_labels.shape.as_list()[1]))
            indices = gcop.reshape(
                argmax_iou.cast(gcop.int32) + gcop.expand_dims(
                    gcop.range(batch_size, dtype=gcop.int32) *
                    local_interval.cast(gcop.int32), 1), [-1]).cast(gcop.int32)
            max_classes = gcop.reshape(
                gcop.gather(gcop.reshape(gt_labels, [-1, 1]), indices, axis=0),
                [batch_size, -1
                 ])
            max_overlap = gcop.reduce_max(iou, axis=2)
            bg_mask = gcop.math.equal(max_overlap,
                                      gcop.zeros_like(max_overlap))
            max_classes = gcop.where(bg_mask, gcop.zeros_like(max_classes),
                                     max_classes)

            max_boxes = gcop.reshape(
                gcop.gather(gcop.reshape(gt_boxes, [-1, 4]), indices, axis=0),
                [batch_size, -1, 4])
            max_boxes = gcop.where(
                gcop.tile(gcop.expand_dims(bg_mask, axis=2), [1, 1, 4]),
                gcop.zeros_like(max_boxes), max_boxes)
        return max_boxes, max_classes, max_overlap, argmax_iou
