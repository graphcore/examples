# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import torch

from utils.tools import ioa, standardize_labels, bbox_iou, sparse_mean


class TestTools:
    def test_ioa_some_intersection(self):
        target_box = np.array([[0, 0, 4, 6]])
        bboxes = np.array(
            [
                [2, 2, 6, 8],
                [1, 1, 8, 10],
            ]
        )
        assert np.array_equal(ioa(target_box, bboxes), np.array([[8 / 24, 15 / 63]]))

    def test_standardize_labels(self):
        labels = np.array([[10.0, 1.0, 1.0, 1.0, 1.0], [14.0, 0.0, 0.0, 0.0, 0.0]])
        width = 4.1
        height = 5.2

        standard_labels = standardize_labels(labels, width, height)

        assert np.array_equal(
            standard_labels, np.array([[10.0, width, height, width, height], [14.0, 0.0, 0.0, 0.0, 0.0]])
        )

    def test_bbox_iou(self):
        boxes1 = torch.Tensor(
            [
                [0.42914, 0.99651, 1.67278, 1.09946],
                [0.56671, 1.19914, 2.57671, 1.80832],
                [0.54428, 0.47646, 1.30271, 1.98252],
                [-0.15353, 1.35289, 0.64380, 2.56266],
            ]
        )
        boxes2 = torch.Tensor(
            [
                [0.25806, 0.10555, 2.00689, 5.62854],
                [0.44211, 0.57287, 4.55000, 8.32778],
                [0.13042, 0.39268, 5.02367, 7.97550],
                [0.79814, 0.10519, 2.45133, 6.58902],
            ]
        )

        actual_ciou = bbox_iou(boxes1, boxes2, is_xyxy=False)
        expected_ciou = torch.Tensor([0.11077, 0.11088, 0.06245, 0.04422])
        assert torch.max(torch.abs(actual_ciou - expected_ciou)) <= 1e-5

        actual_diou = bbox_iou(boxes1, boxes2, is_xyxy=False, special_iou_type="diou")
        expected_diou = torch.Tensor([0.13932, 0.11844, 0.06245, 0.04425])
        assert torch.max(torch.abs(actual_diou - expected_diou)) <= 1e-5

        actual_iou = bbox_iou(boxes1, boxes2, is_xyxy=False, special_iou_type=None)
        expected_iou = torch.Tensor([0.16236, 0.12297, 0.06446, 0.09383])
        assert torch.max(torch.abs(actual_iou - expected_iou)) <= 1e-5

    def test_sparse_mean(self):
        assert sparse_mean(torch.Tensor([1, 2, 3, 0, 0, 0.1, 1e-8])) == torch.Tensor([1.525])
