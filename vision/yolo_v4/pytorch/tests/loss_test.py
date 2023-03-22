# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import copy
import os
import pytest

import torch
import poptorch
from poptorch.optim import SGD

from models.loss import Yolov4_loss
from utils.postprocessing import IPUPredictionsPostProcessing
from tests.test_tools import get_image_and_label, prepare_model, get_cfg, post_process_and_eval


class TestLoss:
    """Tests for end-to-end training and infernece of Yolov4P5."""

    cfg = get_cfg()

    def test_loss_components(self):
        cfg = get_cfg()
        cfg.model.image_size = 128
        cfg.model.ipu = False
        loss = Yolov4_loss(cfg)
        base_path = os.environ["PYTORCH_APPS_DETECTION_PATH"]
        predictions = torch.load(base_path + "/tests/yolo_loss_input/pred_1268_128.pt", map_location="cpu")
        y = torch.load(base_path + "/tests/yolo_loss_input/target_1268_128.pt")

        labels = torch.zeros(1, 1800, 5)
        labels[:, :11, 1:] = y[:, 2:]
        labels[:, :11, 0] = y[:, 1] + 1

        total_loss, lbox, lobj, lcls = loss(predictions, labels)

        assert abs(lbox - 0.079832) <= 1e-3
        assert abs(lcls - 0.070284) <= 1e-3
        assert abs(lobj - 0.425976) <= 1e-3
        assert abs(total_loss - 0.576092) <= 1e-3

        predictions = torch.load(base_path + "/tests/yolo_loss_input/pred_285_128.pt", map_location="cpu")
        y = torch.load(base_path + "/tests/yolo_loss_input/target_285_128.pt")

        labels = torch.zeros(1, 1800, 5)
        labels[:, :1, 1:] = y[:, 2:]
        labels[:, :1, 0] = y[:, 1] + 1

        total_loss, lbox, lobj, lcls = loss(predictions, labels)

        assert abs(lbox - 0.11769) <= 1e-3
        assert abs(lobj - 0.14149) <= 1e-3
        assert abs(lcls - 0.10029) <= 1e-3
        assert abs(total_loss - 0.35946) <= 1e-3

    @pytest.mark.ipus(1)
    @pytest.mark.skip(reason="To be enabled for debugging purposes only, test runs for ~30 minutes")
    def test_loss_standalone(self):
        cfg = get_cfg()
        cfg.inference.nms = False
        cfg.model.mode = "train"
        cfg.model.pipeline_splits = [
            "backbone.cspdark3.csp.res_modules.0",
            "backbone.cspdark5.downsample",
            "neck.cspUp2.bneck_csp.conv2",
        ]
        cfg.ipuopts.gradient_accumulation = 7
        ipu_training_model = prepare_model(cfg)

        cfg.model.ipu = False
        cpu_training_model = prepare_model(cfg)

        transformed_images, transformed_labels, _ = get_image_and_label(cfg)
        transformed_images = transformed_images.repeat([cfg.ipuopts.gradient_accumulation, 1, 1, 1])
        transformed_labels = transformed_labels.repeat([cfg.ipuopts.gradient_accumulation, 1, 1])

        ipu_predictions, (ipu_total_loss, ipu_lbox, ipu_lobj, ipu_lcls) = ipu_training_model(
            transformed_images, transformed_labels
        )
        cpu_total_loss, cpu_lbox, cpu_lobj, cpu_lcls = cpu_training_model.loss(ipu_predictions, transformed_labels)

        assert abs(cpu_total_loss - ipu_total_loss) <= 1e-3
        assert abs(cpu_lbox - ipu_lbox) <= 1e-3
        assert abs(cpu_lobj - ipu_lobj) <= 1e-3
        assert abs(cpu_lcls - ipu_lcls) <= 1e-3

    def test_weight_update(self):
        cfg = get_cfg()
        cfg.inference.nms = False
        cfg.model.mode = "train"
        cfg.model.ipu = False
        cfg.model.image_size = 416
        cfg.model.max_nlabels_p3 = 500
        cfg.model.max_nlabels_p4 = 500
        cfg.model.max_nlabels_p5 = 500
        training_model_1 = prepare_model(cfg)
        optimizer_1 = SGD(training_model_1.parameters(), lr=0.001, momentum=0.9)
        initial_weight = copy.deepcopy(training_model_1.state_dict())

        # use exact number of labels (no padding)
        cfg.model.max_nlabels_p3 = 63
        cfg.model.max_nlabels_p4 = 84
        cfg.model.max_nlabels_p5 = 27
        training_model_2 = prepare_model(cfg)
        training_model_2.load_state_dict(initial_weight)
        optimizer_2 = SGD(training_model_2.parameters(), lr=0.001, momentum=0.9)

        transformed_images, transformed_labels, image_sizes = get_image_and_label(cfg)

        # train for 3 steps then compare the weights
        for i in range(3):
            _, (total_loss_1, lbox_1, lobj_1, lcls_1) = training_model_1(transformed_images, transformed_labels)
            total_loss_1.backward()
            optimizer_1.step()

            _, (total_loss_2, lbox_2, lobj_2, lcls_2) = training_model_2(transformed_images, transformed_labels)
            total_loss_2.backward()
            optimizer_2.step()

        for k, model_1_val in training_model_1.state_dict().items():
            model_2_val = training_model_2.state_dict()[k]
            assert torch.abs(model_1_val - model_2_val).max() <= 6e-4
