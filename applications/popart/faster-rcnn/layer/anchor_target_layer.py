# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# This file has been modified by Hu Di Graphcore Ltd.
# --------------------------------------------------------
import numpy as np
from utils.bbox import bbox_overlaps, bbox_transform_npy, bbox_transform
from config import cfg
import numpy.random as npr
import torch
from utils.random import StateManager
import math
from models.base_model import DetectBase


class AnchorTargetLayer(DetectBase):
    """Assign anchors to ground-truth targets.
    And produces anchor classification labels
    and bounding-box regression targets.
    """

    def __init__(self, feat_stride, scales, ratios, dtype):
        super(AnchorTargetLayer, self).__init__()
        self._feat_stride = feat_stride
        self._scales = scales
        self.dtype = dtype
        self._anchors = torch.from_numpy(
            self.generate_anchors(scales=np.array(scales),
                                  ratios=np.array(ratios))).float()
        # self.anchors = self.Tensor(self.boxes)
        self._num_anchors = self._anchors.shape[0]

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def bbox_overlaps_batch(self, anchors, gt_boxes):
        """
        anchors: (N, 4) ndarray of float
        gt_boxes: (b, K, 5) ndarray of float

        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        batch_size = gt_boxes.size(0)

        #
        if anchors.dim() == 2:

            N = anchors.size(0)
            K = gt_boxes.size(1)

            anchors = anchors.view(1, N, 4).expand(batch_size, N,
                                                   4).contiguous()
            gt_boxes = gt_boxes[:, :, :4].contiguous()

            gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
            gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
            gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

            anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
            anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
            anchors_area = (anchors_boxes_x * anchors_boxes_y).view(
                batch_size, N, 1)

            gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
            anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

            boxes = anchors.view(batch_size, N, 1,
                                 4).expand(batch_size, N, K, 4)
            query_boxes = gt_boxes.view(batch_size, 1, K,
                                        4).expand(batch_size, N, K, 4)

            iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
                  torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
                  torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
            ih[ih < 0] = 0
            ua = anchors_area + gt_boxes_area - (iw * ih)
            overlaps = iw * ih / ua

            # mask the overlap here.
            overlaps.masked_fill_(
                gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K),
                0)
            overlaps.masked_fill_(
                anchors_area_zero.view(batch_size, N,
                                       1).expand(batch_size, N, K), -1)

        elif anchors.dim() == 3:
            N = anchors.size(1)
            K = gt_boxes.size(1)

            if anchors.size(2) == 4:
                anchors = anchors[:, :, :4].contiguous()
            else:
                anchors = anchors[:, :, 1:5].contiguous()

            gt_boxes = gt_boxes[:, :, :4].contiguous()

            gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
            gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
            gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

            anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
            anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
            anchors_area = (anchors_boxes_x * anchors_boxes_y).view(
                batch_size, N, 1)

            gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
            anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

            boxes = anchors.view(batch_size, N, 1,
                                 4).expand(batch_size, N, K, 4)
            query_boxes = gt_boxes.view(batch_size, 1, K,
                                        4).expand(batch_size, N, K, 4)

            iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
                  torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
                  torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
            ih[ih < 0] = 0
            ua = anchors_area + gt_boxes_area - (iw * ih)

            overlaps = iw * ih / ua

            # mask the overlap here.
            overlaps.masked_fill_(
                gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K),
                0)
            overlaps.masked_fill_(
                anchors_area_zero.view(batch_size, N,
                                       1).expand(batch_size, N, K), -1)
        else:
            raise ValueError('anchors input dimension is not correct.')

        return overlaps

    def change2tensor(self, rpn_data):
        rpn_label = rpn_data[0].view(1, -1)
        rpn_keep = rpn_label.view(-1).ne(-1).nonzero().view(-1).numpy()
        rpn_label = rpn_label.numpy()
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[
            1:]
        rpn_bbox_targets = rpn_bbox_targets.numpy()
        rpn_bbox_inside_weights = rpn_bbox_inside_weights.numpy()
        rpn_bbox_outside_weights = rpn_bbox_outside_weights.numpy()

        return rpn_label, rpn_keep, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def forward(self, gt_boxes, im_info, num_boxes):
        Batch_Size = gt_boxes.shape[0]
        rpn_label, rpn_keep, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = [], [], [], [], []
        for idx in range(Batch_Size):
            _num_boxes = num_boxes[idx]
            _rpn_label, _rpn_keep, _rpn_bbox_targets, _rpn_bbox_inside_weights, _rpn_bbox_outside_weights = self.change2tensor(
                self._forward([
                    torch.from_numpy(gt_boxes[idx:idx + 1, :_num_boxes, :]),
                    im_info,
                ]))
            rpn_label.append(_rpn_label)
            rpn_keep.append(_rpn_keep[np.newaxis, :])
            rpn_bbox_targets.append(_rpn_bbox_targets)
            rpn_bbox_inside_weights.append(_rpn_bbox_inside_weights)
            rpn_bbox_outside_weights.append(_rpn_bbox_outside_weights)
        rpn_label = np.concatenate(rpn_label, axis=0)
        rpn_keep = np.concatenate(rpn_keep, axis=0)
        rpn_bbox_targets = np.concatenate(rpn_bbox_targets, axis=0)
        rpn_bbox_inside_weights = np.concatenate(rpn_bbox_inside_weights,
                                                 axis=0)
        rpn_bbox_outside_weights = np.concatenate(rpn_bbox_outside_weights,
                                                  axis=0)
        return rpn_label, rpn_keep, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def _forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # raise Exception('not debugged')
        gt_boxes = input[0]
        im_info = input[1]
        feat_width, feat_height = math.ceil(
            cfg.INPUT_SIZE[0] / cfg.FEAT_STRIDE), math.ceil(
                cfg.INPUT_SIZE[1] / cfg.FEAT_STRIDE)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(
            np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                       shift_y.ravel())).transpose())
        shifts = shifts.contiguous().float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(
            gt_boxes)
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4).detach().cpu().numpy()
        gt_boxes = gt_boxes.detach().cpu().numpy()

        assert gt_boxes.shape[0] == 1
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(
            feat_height, feat_width, gt_boxes[0], im_info, all_anchors,
            self._num_anchors)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform_npy(ex_rois, gt_rois[:, :4])


@StateManager()
def anchor_target_layer(feat_height, feat_width, gt_boxes, im_info,
                        all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    height, width = feat_height, feat_width

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -_allowed_border) & (all_anchors[:, 1] >= -_allowed_border) & (
        all_anchors[:, 2] < im_info[1] + _allowed_border) & (all_anchors[:, 3] < im_info[0] + _allowed_border))[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                             np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds,
                                  size=(len(fg_inds) - num_fg),
                                  replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds,
                                  size=(len(bg_inds) - num_bg),
                                  replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(
        cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights,
                                 total_anchors,
                                 inds_inside,
                                 fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights,
                                  total_anchors,
                                  inds_inside,
                                  fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights

    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = [
        torch.from_numpy(ele) for ele in [
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights,
            rpn_bbox_outside_weights
        ]
    ]
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def unmap(data, count, inds, fill=0):
    """
    Unmap a subset of item (data) back to the original
    set of items (ofsize count)
    """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def bbox_transform_batch(ex_rois, gt_rois):
    #
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        #
        targets_dx = (gt_ctr_x -
                      ex_ctr_x.view(1, -1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y -
                      ex_ctr_y.view(1, -1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths /
                               ex_widths.view(1, -1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights /
                               ex_heights.view(1, -1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        #
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 2)
    #
    return targets
