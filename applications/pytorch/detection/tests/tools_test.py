# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import torch

from utils.tools import ioa, nms, standardize_labels, xyxy_to_xywh


class TestTools:

    def test_ioa_some_intersection(self):
        target_box = np.array([[0, 0, 4, 6]])
        bboxes = np.array([
            [2, 2, 6, 8],
            [1, 1, 8, 10],
        ])
        assert np.array_equal(
            ioa(target_box, bboxes), np.array([[8/24, 15/63]])
        )

    def test_nms(self):
        boxes = torch.Tensor([[
            [0, 0, 4.1, 6.1],
            [0, 0, 4, 6],
            [7, 9, 20, 25]
        ]])
        obj_scores = torch.Tensor([[[0.9], [1.0], [0.1]]])
        cls_scores = torch.Tensor([[[0.9, 0.1, 0.1], [0.9, 0.2, 0.3], [0.1, 0.1, 0.2]]])

        batch_size = boxes.shape[1]
        scores = obj_scores * cls_scores
        classes = torch.arange(batch_size).view(boxes.shape[0], 1, cls_scores.shape[1]).repeat(1, batch_size, 1)
        boxes = boxes.repeat(1, 1, cls_scores.shape[1]).view(batch_size, -1, boxes.shape[-1])

        pruned = nms(scores.view(batch_size, -1), boxes, classes.view(batch_size, -1), iou_threshold = 0.5, max_detections = 6)

        score_pos = 1
        box_pos = 2
        class_pos = 3
        score_value = pruned[score_pos][1][0].unsqueeze(axis=-1)
        box_value = pruned[box_pos][1][0]
        class_value = pruned[class_pos][1][0].unsqueeze(axis=-1)

        result = torch.cat((box_value, score_value, class_value), axis=-1).float()

        assert torch.equal(
            result, torch.Tensor([0, 0, 4, 6, 0.9, 0]).float()
        )

    def test_standardize_labels(self):
        labels = np.array([
            [10.0, 1.0, 1.0, 1.0, 1.0],
            [14.0, 0.0, 0.0, 0.0, 0.0]
        ])
        width = 4.1
        height = 5.2

        standard_labels = standardize_labels(labels, width, height)

        assert np.array_equal(
            standard_labels, np.array([[10.0, width, height, width, height], [14.0, 0.0, 0.0, 0.0, 0.0]])
        )
