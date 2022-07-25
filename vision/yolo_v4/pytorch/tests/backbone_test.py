# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import numpy as np
import torch
import torch.nn as nn

from models.backbone.yolov4_p5 import Yolov4P5BackBone
from poptorch import inferenceModel


class TestYolov4P5BackBone:
    """Tests inference of Yolov4P5 backbone."""

    @pytest.mark.ipus(1)
    def test_inference(self):
        model = Yolov4P5BackBone(3, nn.ReLU())
        model = inferenceModel(model.half())
        y = model(torch.Tensor(np.random.randn(1, 3, 64, 64)))
        assert y[0].shape == torch.Size([1, 1024, 2, 2])
        assert y[1].shape == torch.Size([1, 512, 4, 4])
        assert y[2].shape == torch.Size([1, 256, 8, 8])
