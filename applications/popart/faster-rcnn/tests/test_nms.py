# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di

import numpy as np
import sys
import os
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../IPU'))
from ipu_tensor import gcop
from gc_session import Session

# make test data for NMS
IMAGE_SIZE = [800, 800]
BOX_SIZE_RANGE = [5, 500]
IN_BOX_NUM = 18000
OUT_BOX_NUM = 2000
DTYPE = np.float32
SCORE_THRD = 0.0


def helper_func(iou_thrd):
    box_centers = np.random.rand(1, IN_BOX_NUM, 2) * np.asarray(IMAGE_SIZE)
    box_whs = np.random.rand(
        1, IN_BOX_NUM, 2) * (BOX_SIZE_RANGE[1] - BOX_SIZE_RANGE[0]) + BOX_SIZE_RANGE[0]
    start_xys = box_centers - box_whs / 2
    end_xys = box_centers + box_whs / 2
    boxes = np.concatenate([start_xys, end_xys], axis=2).astype(DTYPE)
    scores = np.random.rand(1, IN_BOX_NUM).astype(DTYPE)

    # define helper function
    def clip_boxes(boxes, im_info):
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

    def py_cpu_nms(dets, scores, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = scores  # bbox打分

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return np.asarray(keep)

    # build net
    with gcop.device("0"):
        input_boxes = gcop.placeholder(shape=[1, IN_BOX_NUM, 4], dtype=DTYPE)
        input_scores = gcop.placeholder(shape=[1, IN_BOX_NUM], dtype=DTYPE)
        clipped_boxes = clip_boxes(input_boxes, IMAGE_SIZE)
        output_boxes, output_keep, num_valids = gcop.cOps.nms(
            input_scores,
            input_boxes=clipped_boxes,
            threshold=iou_thrd,
            numDetections=OUT_BOX_NUM,
            score_threshold=SCORE_THRD)

    # make session
    gcop.safe_mode_on()
    sess = Session([output_boxes, output_keep, num_valids, clipped_boxes])

    # run session
    feed_dict = {input_boxes: boxes, input_scores: scores}
    output_boxes, output_keep, num_valids, clipped_boxes = sess.run(feed_dict)
    num_valids = num_valids.data[0]
    ipu_keeps = list(
        filter(lambda x: x > -1, output_keep.data.flatten().tolist()))
    ipu_boxes = clipped_boxes.data[:, ipu_keeps, :]
    ipu_scores = scores[:, ipu_keeps]

    # run reference
    keeps = py_cpu_nms(ipu_boxes[0], ipu_scores[0], iou_thrd)
    org_keeps = py_cpu_nms(boxes[0], scores[0], iou_thrd)[:OUT_BOX_NUM]

    # compare result
    # Ensure that basically all overlapped boxes are filtered out
    assert len(keeps)/len(ipu_keeps) > 0.99
    # Ensure that most of the non-overlapping boxes are retained
    assert len(ipu_keeps)/len(org_keeps) > 0.8 and len(ipu_keeps) / \
        len(org_keeps) < 1.2
    # Ensure that the output boxes has not been modified
    assert np.all(ipu_boxes == output_boxes.data[:, :num_valids, :])


def test_nms():
    faster_rcnn_working_dic = os.path.join(os.path.dirname(__file__), '../')
    subprocess.run(['make'], shell=True, cwd=faster_rcnn_working_dic)
    for iou_thrd in [0.1, 0.3, 0.5, 0.7, 0.9]:
        helper_func(iou_thrd)
