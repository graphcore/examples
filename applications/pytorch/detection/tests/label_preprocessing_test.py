# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest

import torch
import numpy as np
import torch.nn as nn

from utils.config import get_cfg_defaults
from models.yolov4_p5 import PreprocessTargets
from poptorch import inferenceModel


class TestYolov4P5Preprocessing:
    """Tests for Yolov4loss labels preprocessing."""
    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_preprocessing(self):
        config = get_cfg_defaults()

        config.model.strides = [8, 16, 32]
        config.model.n_classes = 80
        config.model.image_size = 640
        config.model.input_channels = 3
        config.freeze()

        model = PreprocessTargets(config.model)
        model = inferenceModel(model.half())

        x = torch.zeros((1, 90, 5))
        real_labels = torch.tensor([[45.0000,  0.4795,  0.6416,  0.9556,  0.4466], [45.0000,  0.7365,  0.3104,  0.4989,  0.3573]])
        x[:, :real_labels.shape[0]] = real_labels

        t_indices_boxes_p5, t_indices_boxes_p4, t_indices_boxes_p3 = model(x)
        anchor_ind_p5, yind_p5, xind_p5, t_boxes_p5 = t_indices_boxes_p5
        anchor_ind_p4, yind_p4, xind_p4, t_boxes_p4 = t_indices_boxes_p4
        anchor_ind_p3, yind_p3, xind_p3, t_boxes_p3 = t_indices_boxes_p3

        n_digits = 3
        t_boxes_p4 = torch.round(t_boxes_p4 * 10**n_digits) / (10**n_digits)
        t_boxes_p3 = torch.round(t_boxes_p3 * 10**n_digits) / (10**n_digits)

        size = 5 * len(config.model.anchors.p3width) * config.dataset.max_bbox_per_scale
        assert anchor_ind_p5.shape == torch.Size([size])
        assert yind_p4.shape == torch.Size([size])
        assert xind_p3.shape == torch.Size([size])
        assert t_boxes_p5.shape == torch.Size([size, 5])
        assert xind_p5[xind_p5 != 0.].numel() == 0

        _, index_t_boxes_p4 = torch.sort(t_boxes_p4.sum(axis=-1))
        _, index_t_boxes_p3 = torch.sort(t_boxes_p3.sum(axis=-1))
        t_boxes_p4 = t_boxes_p4[index_t_boxes_p4]
        t_boxes_p3 = t_boxes_p3[index_t_boxes_p3]

        assert torch.all(torch.eq(anchor_ind_p4[anchor_ind_p4 != 0.].sum(), torch.tensor([3, 3, 3, 1, 3, 2, 2, 3, 2, 1, 3, 1]).sum()))
        assert torch.all(torch.eq(yind_p3[yind_p3 != 0.].sum(), torch.tensor([13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12,  6,  6,  6,  6,  6,  6, 6,  6,  5,  5,  5,  5]).sum()))
        assert t_boxes_p5[t_boxes_p5 != 0.].numel() == 0
        assert torch.all(torch.eq(t_boxes_p4[t_boxes_p4 != 0.].view(-1, 5)[-2:], torch.tensor([[45.0000,  0.1800,  0.6640, 38.2240, 17.8640],
                                                                                               [45.0000, 1.1800,  0.6640, 38.2240, 17.8640]])))
        assert torch.all(torch.eq(t_boxes_p3[t_boxes_p3 != 0.].view(-1, 5)[-2:], torch.tensor([[45.0000,  0.5900, 0.8320, 19.1120,  8.9320],
                                                                                               [45.0000,  0.5900,  0.8320, 19.1120,  8.9320]])))
