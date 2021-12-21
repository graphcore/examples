# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import ctypes
import os

import torch
import poptorch

from utils.custom_ops import Nms
from utils.config import get_cfg_defaults


class TestNms:
    """Tests nms custom op"""
    def test_nms(self):
        cfg_inference = get_cfg_defaults().inference
        cfg_inference.class_conf_threshold = 0.1
        cfg_inference.iou_threshold = 0.1
        cfg_inference.nms_max_detections = 6
        batch_size = 2
        n_elements = 6
        n_classes = 2

        ipu_model = Nms(cfg_inference, cpu_mode=False)
        cpu_model = Nms(cfg_inference, cpu_mode=True)

        ipu_model = poptorch.inferenceModel(ipu_model)

        scores = torch.tensor([.9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7,
                               .9,  .1,  .25, .75, .4, .6, .95, .05, .5,  .5,  .3, .7], dtype=torch.float32).reshape(batch_size, n_elements, n_classes)
        boxes = torch.tensor([0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11, 0, 10.1, 1, 11.1,
                              0, 100, 1, 101, 0, 0, 1, 1, 0, 0.1, 1, 1.1, 0, -0.1, 1, 0.9, 0, 10, 1, 11,
                              0, 10.1, 1, 11.1, 0, 100,  1, 101], dtype=torch.float32).reshape(batch_size, n_elements, 4)

        res_ipu = ipu_model(scores, boxes)

        classes = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.int32)

        sorted_idx, _ = torch.arange(0, boxes.shape[1], 1).repeat(n_classes).sort()
        boxes = boxes.index_select(1, sorted_idx)

        shifting = 4.
        box_shift = (classes.float() * shifting).unsqueeze(axis=-1).float()
        shifted_box = boxes.view(batch_size, n_elements * n_classes, 4) + box_shift
        res_cpu = cpu_model(scores.view(2, -1), shifted_box, classes)
        box_shift = (res_cpu[3].float() * shifting).unsqueeze(axis=-1).float()

        assert torch.equal(res_ipu[0].long(), res_cpu[0] // n_classes)
        assert torch.equal(res_ipu[1], res_cpu[1])
        assert torch.all(res_ipu[2] - (res_cpu[2] - box_shift) < 2e-07)
        assert torch.equal(res_ipu[3], res_cpu[3])
        assert torch.equal(res_ipu[4], res_cpu[4])
