# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import numpy as np
import torch

from utils.tools import ioa, nms, standardize_labels, xyxy_to_xywh


class TestTools:

    @pytest.mark.category1
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
        boxes = torch.Tensor([
            [0, 0, 4.1, 6.1],
            [0, 0, 4, 6],
            [7, 9, 20, 25]
        ])
        obj_scores = torch.Tensor([[0.9], [1.0], [0.1]])
        cls_scores = torch.Tensor([[0.9, 0.1, 0.1], [0.9, 0.2, 0.3], [0.1, 0.1, 0.2]])
        predictions = torch.unsqueeze(torch.cat((boxes, obj_scores, cls_scores), axis=1), 0)

        pruned = nms(predictions, iou_threshold = 0.5, score_threshold = 0.5)[0]
        pruned = xyxy_to_xywh(pruned)
        assert torch.equal(
            pruned.float(), torch.Tensor([[0, 0, 4, 6, 0.9, 0]]).float()
        )

    @pytest.mark.category1
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
