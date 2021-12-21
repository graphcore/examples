# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import math
import torch
import numpy as np
from collections import OrderedDict
from IPU.ipu_tensor import gcop
from IPU.gc_session import add_aux_vars_to_tensor
from config import cfg


class BaseModel:
    inputs = OrderedDict()
    outputs = OrderedDict()
    loss = None

    def __init__(self, fp16_on=False, training=True):
        self.dtype = np.float16 if fp16_on else np.float32
        self.fp16_on = fp16_on
        self.training = training
        if self.dtype == np.float32:
            self.gcType = gcop.float32
        elif self.dtype == np.float16:
            self.gcType = gcop.float16

    def add_input(self, shape=None, dtype=np.float32, name=None):
        input_tensor = gcop.placeholder(shape=shape, dtype=dtype, name=name)
        if name is None or name == '':
            name = input_tensor.name
        assert name not in BaseModel.inputs
        BaseModel.inputs[name] = input_tensor
        return input_tensor

    def add_output(self,
                   idx_name='',
                   output=None,
                   gradient_on=False,
                   other_aux_vars=[]):
        assert output is not None
        output_name = output.name
        if idx_name == '':
            idx_name = output_name

        if idx_name in BaseModel.outputs:
            assert BaseModel.outputs[
                idx_name].name == output_name, 'add one idx_name twice, and the target tensor is not same'
        else:
            BaseModel.outputs[idx_name] = output

        prefixes = []
        if gradient_on:
            prefixes.append('Gradient___')

        for aux_var in other_aux_vars:
            prefixes.append(aux_var)

        if len(prefixes) > 0:
            add_aux_vars_to_tensor(output, prefixes)

    def record_all_weights_grad(self, ):
        all_weights = gcop.trainable_variables()
        for weight in all_weights:
            self.add_output(output=weight, gradient_on=True)
        print('all weights grad recorded!!!')

    def normal_init(self, shape, mean, stddev, truncated=False, dtype=np.float32):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        weight = torch.zeros(shape, dtype=torch.float32)
        if truncated:
            weight.data.normal_().fmod_(2).mul_(stddev).add_(
                mean)  # not a perfect approximation
        else:
            weight.data.normal_(mean, stddev)
        return weight.numpy().astype(dtype)


def smooth_l1_loss(bbox_pred,
                   bbox_targets,
                   bbox_inside_weights,
                   bbox_outside_weights,
                   sigma=1.0,
                   reduceDim=None,
                   debugPrefix=''):
    """SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                    |x| - 0.5 / sigma^2,    otherwise
    """

    sigma2 = sigma * sigma  # 1.0
    #
    if bbox_inside_weights is None:
        inside_mul = bbox_pred - bbox_targets
    else:
        inside_sub = bbox_pred - bbox_targets
        inside_mul = bbox_inside_weights * inside_sub

    dst_type = inside_mul.dtype
    smooth_l1_sign = gcop.less(
        gcop.abs(inside_mul), gcop.constant(np.asarray(1.0 / sigma2),
                                            dst_type))
    smooth_l1_sign = smooth_l1_sign.cast(inside_mul.dtype).detach()

    smooth_l1_option1 = inside_mul * inside_mul * gcop.constant(
        np.asarray(0.5 * sigma2), dst_type)

    smooth_l1_option2 = gcop.abs(inside_mul) - gcop.constant(
        np.asarray(0.5 / sigma2), dst_type)

    smooth_l1_result = smooth_l1_option1 * smooth_l1_sign + smooth_l1_option2 * (
        gcop.abs(smooth_l1_sign - gcop.constant(np.asarray(1.0), dst_type)))

    if bbox_outside_weights is None:
        outside_mul = smooth_l1_result
    else:
        outside_mul = bbox_outside_weights * smooth_l1_result

    with gcop.variable_scope(debugPrefix):
        outside_mul = gcop.reduce_sum(outside_mul, reduceDim, keepdims=0)
        rest_dims = list(range(len(outside_mul.shape.as_list())))
        if len(rest_dims) > 0:
            outside_mul = gcop.reduce_mean(outside_mul, rest_dims, keepdims=0)
    return outside_mul


def bbox_overlaps_tf(anchors, gt_boxes):
    #
    gt_boxes = gt_boxes[:, :, :4]

    bb_y_min = anchors[:, :, 0]
    bb_x_min = anchors[:, :, 1]
    bb_y_max = anchors[:, :, 2]
    bb_x_max = anchors[:, :, 3]

    gt_y_min = gt_boxes[:, :, 0]
    gt_x_min = gt_boxes[:, :, 1]
    gt_y_max = gt_boxes[:, :, 2]
    gt_x_max = gt_boxes[:, :, 3]

    #
    i_xmin = gcop.maximum([bb_x_min, gcop.transpose(gt_x_min, [0, 2, 1])])
    i_xmax = gcop.minimum([bb_x_max, gcop.transpose(gt_x_max, [0, 2, 1])])
    i_ymin = gcop.maximum([bb_y_min, gcop.transpose(gt_y_min, [0, 2, 1])])
    i_ymax = gcop.minimum([bb_y_max, gcop.transpose(gt_y_max, [0, 2, 1])])
    i_area = (gcop.maximum(
        i_xmax - i_xmin, gcop.constant(np.asarray([0]).astype(
            self.dtype)))) * (gcop.maximum(
                i_ymax - i_ymin,
                gcop.constant(np.asarray([0]).astype(self.dtype))))

    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)

    u_area = bb_area + gcop.transpose(gt_area,
                                      [0, 2, 1]) - i_area + gcop.constant(
                                          np.asarray(1e-6).astype(self.dtype))

    iou = i_area / u_area

    padding_mask = gcop.less(i_xmin,
                             gcop.constant(np.asarray([0]).astype(self.dtype)))

    iou = gcop.where(padding_mask,
                     gcop.constant(np.asarray([-1]).astype(self.dtype)), iou)

    return iou


def bbox_overlaps_torch(boxes, query_boxes):
    """
    reference: https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/utils/bbox.py
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
        (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
        (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (gcop.minimum(boxes[:, 2:3], query_boxes[:, 2].unsqueeze(0)) -
          gcop.maximum(boxes[:, 0:1], query_boxes[:, 0].unsqueeze(0)) + 1)
    iw = gcop.clip_by_value(iw, clip_value_max=np.inf, clip_value_min=0.)
    ih = (gcop.minimum(boxes[:, 3:4], query_boxes[:, 3].unsqueeze(0)) -
          gcop.maximum(boxes[:, 1:2], query_boxes[:, 1].unsqueeze(0)) + 1)
    ih = gcop.clip_by_value(ih, clip_value_max=np.inf, clip_value_min=0.)
    ua = box_areas.reshape([-1, 1]) + query_areas.reshape([1, -1]) - iw * ih
    overlaps = iw * ih / ua
    return overlaps


def get_valid_area_mask(boxes):
    # input boxes: 1,n,4
    # output: mask: n,1
    ws = boxes[:, :, 2] - boxes[:, :, 0]  # 1,n
    hs = boxes[:, :, 3] - boxes[:, :, 1]  # 1,n
    areas = ws * hs  # 1,n
    valid_flags = gcop.greater(areas, 0.0)
    valid_mask = gcop.cast(valid_flags, boxes.dtype)  # 1,n
    return gcop.transpose(valid_mask, [1, 0])


class DetectBase:
    def _whctrs(self, anchor):
        """Return width, height, x center, and y center for an anchor .
        """
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def _mkanchors(self, ws, hs, x_ctr, y_ctr):
        """Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
        return anchors

    def _ratio_enum(self, anchor, ratios):
        """Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        size = w * h
        size_ratios = size // ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _scale_enum(self, anchor, scales):
        """Enumerate a set of anchors for each scale wrt an anchor.
        """
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def generate_anchors(self,
                         base_size=16,
                         ratios=[0.5, 1, 2],
                         scales=2**np.arange(3, 6)):
        """Generate anchor by enumerating aspect ratios X
        scales wrt a reference (0, 0, 15, 15) window.
        """
        if cfg.ANCHOR_VERSION == 'v1':
            base_anchor = np.array([1, 1, base_size, base_size]) - 1
            ratio_anchors = self._ratio_enum(base_anchor, ratios)
            anchors = np.vstack([
                self._scale_enum(ratio_anchors[i, :], scales)
                for i in range(ratio_anchors.shape[0])
            ])
            return anchors
        elif cfg.ANCHOR_VERSION == 'v2':
            sizes = [s * base_size for s in scales]
            aspect_ratios = ratios
            return self.generate_anchors_v2(sizes=sizes, aspect_ratios=aspect_ratios)
        else:
            raise NotImplemented

    def generate_anchors_v2(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return np.asarray(anchors)

    def process(self, boxes, feat_size=[30, 48], feat_stride=16):

        feat_height, feat_width = feat_size
        shift_x = np.arange(0, feat_width) * feat_stride
        shift_y = np.arange(0, feat_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel())).transpose()
        A = boxes.shape[0]
        K = shifts.shape[0]
        boxes = boxes.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
        boxes = boxes.reshape(1, K * A, 4)

        return boxes


def nms(input_scores,
        input_boxes,
        threshold=0.7,
        numDetections=300,
        score_threshold=None,
        debugContext=None):

    if debugContext is None:
        debugContext = ''
    output_boxes, output_keep, num_valids = gcop.cOps.nms(
        input_scores,
        input_boxes=input_boxes,
        threshold=threshold,
        numDetections=numDetections,
        score_threshold=score_threshold,
        debugContext=debugContext)

    return output_boxes, output_keep, num_valids


def roi_align(bottom_data,
              bottom_rois,
              spatial_scale=1 / 16.0,
              num_rois=300,
              aligned_height=7,
              aligned_width=7,
              fp16_on=None):

    result = gcop.cOps.roi_align(bottom_data,
                                 bottom_rois,
                                 spatial_scale=spatial_scale,
                                 num_rois=num_rois,
                                 fp16_on=fp16_on,
                                 aligned_height=aligned_height,
                                 aligned_width=aligned_width)

    return result
