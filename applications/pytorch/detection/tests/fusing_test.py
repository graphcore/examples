# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.backbone.yolov4_p5 import CrossStagePartialBlock
from models.detector import Detector
from models.layers import ConvNormAct
from models.neck.yolov4_p5 import CSPSPP
from poptorch import inferenceModel


class TestDetector:
    """Tests weight fusing of conv and batch norm."""
    class MockDetector(Detector):
        def __init__(self, backbone, neck=nn.Identity, detector_head=nn.Identity):
            super().__init__(backbone, neck, detector_head)
            self.neck = nn.Identity()
            self.detector_head = nn.Identity()

        def forward(self, x):
            x = self.backbone(x)
            return x

        def output_shape(self, input_shape):
            return super().output_shape(input_shape)

    def test_fuse_conv(self):
        input_tensor = torch.Tensor(np.random.randint(0, 255, size=(1, 16, 32, 32)))
        model = self.MockDetector(ConvNormAct)
        model.backbone = ConvNormAct(16, 32, norm ='batch')
        model_no_fuse = inferenceModel(model.eval())
        output_no_fuse = model_no_fuse(input_tensor)

        model.optimize_for_inference()
        model_fuse = inferenceModel(model.eval())
        output_fuse = model_fuse(input_tensor)
        assert torch.max(torch.abs(output_no_fuse - output_fuse)) <= 1e-04

    def test_fuse_csp(self):
        input_tensor = torch.Tensor(np.random.randint(0, 255, size=(1, 16, 32, 32)))
        model = self.MockDetector(CrossStagePartialBlock)
        model.backbone = CrossStagePartialBlock(16, 32, norm ='batch')
        model_no_fuse = inferenceModel(model.eval())
        output_no_fuse = model_no_fuse(input_tensor)

        model.optimize_for_inference()
        model_fuse = inferenceModel(model.eval())
        output_fuse = model_fuse(input_tensor)
        assert torch.max(torch.abs(output_no_fuse - output_fuse)) <= 1e-04
