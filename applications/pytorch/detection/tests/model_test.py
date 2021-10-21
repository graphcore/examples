# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from models.yolov4_p5 import Yolov4P5
import numpy as np
from poptorch import inferenceModel, trainingModel
import pytest
import torch
from utils.config import get_cfg_defaults


class TestYolov4P5:
    """Tests for end-to-end training and infernece of Yolov4P5."""

    cfg = get_cfg_defaults()
    cfg.model.image_size = 64

    input_tensor = torch.Tensor(np.random.rand(1, 3, 64, 64))

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    @pytest.mark.skip(reason="to enable when loss is implemented")
    def test_training(self):
        model = Yolov4P5(self.cfg.model)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=False)

        model = trainingModel(model.half(), optimizer=optimizer)
        loss = model(self.input_tensor)
        # TODO implement test when loss is implemented.

    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_inference(self):
        model = Yolov4P5(self.cfg.model)
        model = inferenceModel(model.half().eval())
        y = model(self.input_tensor)

        expected_output_size = model.output_shape((64, 64))

        p3 = expected_output_size['p3']
        p4 = expected_output_size['p4']
        p5 = expected_output_size['p5']

        assert y[0].shape == torch.Size([p3[0],  p3[1] * p3[2] * p3[3], p3[4]])
        assert y[1].shape == torch.Size([p4[0],  p4[1] * p4[2] * p4[3], p4[4]])
        assert y[2].shape == torch.Size([p5[0],  p5[1] * p5[2] * p5[3], p5[4]])

    @pytest.mark.ipus(1)
    def test_inference_cpu_and_ipu(self):
        self.cfg.model.half = False
        self.cfg.model.image_size = 416

        # Create CPU model
        torch.manual_seed(0)
        self.cfg.model.ipu = False
        model = Yolov4P5(self.cfg.model)
        cpu_model = model.eval()
        y_cpu = cpu_model(self.input_tensor)

        # Create IPU model
        torch.manual_seed(0)
        self.cfg.model.ipu = True
        model = Yolov4P5(self.cfg.model)
        ipu_model = inferenceModel(model.eval())
        y_ipu = ipu_model(self.input_tensor)

        assert torch.max(torch.abs(y_cpu[0] - y_ipu[0])) <= 0.002
        assert torch.max(torch.abs(y_cpu[1] - y_ipu[1])) <= 0.002
        assert torch.max(torch.abs(y_cpu[2] - y_ipu[2])) <= 0.002

    def test_fused_inference(self):
        self.cfg.model.normalization = 'batch'
        self.cfg.model.half = False

        model = Yolov4P5(self.cfg.model)
        before_fuse_model = inferenceModel(model.eval())
        before_fuse_output = before_fuse_model(self.input_tensor)

        model.optimize_for_inference()
        after_fuse_model = inferenceModel(model.eval())
        after_fuse_output = after_fuse_model(self.input_tensor)

        assert torch.max(torch.abs(after_fuse_output[0] - before_fuse_output[0])) <= 1e-4
        assert torch.max(torch.abs(after_fuse_output[1] - before_fuse_output[1])) <= 1e-4
        assert torch.max(torch.abs(after_fuse_output[2] - before_fuse_output[2])) <= 1e-4
