# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Inference test that runs a forward pass and compares results with a GPU reference.

import argparse
import io
import numpy as np
import os
from PIL import Image
import pytest
import requests
import time
import yacs

import torch
import poptorch
from poptorch import inferenceModel

from models.detector import Detector
from models.yolov4_p5 import Yolov4P5
from utils.config import get_cfg_defaults
from utils.preprocessing import ResizeImage, Pad, ToTensor
from utils.tools import load_and_fuse_pretrained_weights, post_processing, StatRecorder


def get_cfg():
    cfg = get_cfg_defaults()

    cfg.model.image_size = 896
    cfg.inference.nms = True
    cfg.inference.class_conf_threshold = 0.001
    cfg.inference.iou_threshold = 0.65
    cfg.inference.nms_max_detections = 220
    cfg.inference.pre_nms_topk_k = 1180
    cfg.ipuopts.batches_per_step = 1
    cfg.model.normalization = "batch"
    cfg.model.activation = "mish"
    cfg.model.half = True
    cfg.model.uint_io = True
    cfg.model.input_channels = 3
    cfg.model.micro_batch_size = 1
    cfg.model.mode = "test"
    cfg.model.ipu = True

    return cfg


def ipu_options(opt: argparse.ArgumentParser, cfg: yacs.config.CfgNode, model: Detector):
    """Configurate the IPU options using cfg and opt options.
        Parameters:
            opt: opt object containing options introduced in the command line
            cfg: yacs object containing the config
            model[Detector]: a torch Detector Model
        Returns:
            ipu_opts: Options for the IPU configuration
    """
    batches_per_step = cfg.ipuopts.batches_per_step
    half = cfg.model.half

    ipu_opts = poptorch.Options()
    ipu_opts.deviceIterations(batches_per_step)
    ipu_opts.autoRoundNumIPUs(True)

    if half:
        ipu_opts.Precision.setPartialsType(torch.float16)
        model.half()

    return ipu_opts


def get_image_and_label(cfg):
    url_sample_image = 'http://images.cocodataset.org/val2017/000000100238.jpg'
    img_data = requests.get(url_sample_image).content
    image = Image.open(io.BytesIO(img_data)).convert('RGB')

    height, width = image.size
    image_sizes = torch.Tensor([[height, width]])

    label = np.array([[39, 0.319508, 0.745573, 0.020516, 0.028479],
                      [0, 0.484391, 0.583271, 0.360031, 0.833458],
                      [0, 0.685664, 0.494917, 0.284422, 0.986458],
                      [0, 0.869086, 0.720719, 0.207766, 0.549563],
                      [0, 0.168453, 0.526521, 0.333531, 0.914208],
                      [29, 0.166422, 0.562135, 0.118313, 0.139687],
                      [29, 0.480703, 0.565990, 0.135906, 0.120813],
                      [26, 0.591977, 0.203583, 0.045234, 0.121958],
                      [26, 0.349672, 0.619479, 0.150000, 0.568833],
                      [29, 0.708734, 0.284302, 0.118188, 0.159854]])

    resizer = ResizeImage(cfg.model.image_size)
    padder = Pad(cfg.model.image_size)
    to_tensor = ToTensor(int(cfg.dataset.max_bbox_per_scale), "uint")

    item = (image, label)
    item = resizer(item)
    image, label = padder(item)
    image, label = to_tensor((np.array(image), label))

    return image.unsqueeze(axis=0), label.unsqueeze(axis=0), image_sizes


def prepare_model(cfg):
    opt = argparse.ArgumentParser()
    opt.weights = 'weights/yolov4-p5-sd.pt'

    model = Yolov4P5(cfg)
    model.eval()
    model = load_and_fuse_pretrained_weights(model, opt)
    model.optimize_for_inference()

    ipu_opts = ipu_options(opt, cfg, model)
    return inferenceModel(model, ipu_opts)


def post_process_and_eval(cfg, y, image_sizes, transformed_labels):
    stat_recorder = StatRecorder(cfg)
    processed_batch = post_processing(cfg, y, image_sizes, transformed_labels)

    pruned_preds_batch = processed_batch[0]
    processed_labels_batch = processed_batch[1]
    for idx, (pruned_preds, processed_labels) in enumerate(zip(pruned_preds_batch, processed_labels_batch)):
        stat_recorder.record_eval_stats(processed_labels, pruned_preds, image_sizes[idx])

    return stat_recorder.compute_and_print_eval_metrics(print)


class TestYolov4P5:
    """Tests for end-to-end training and infernece of Yolov4P5."""

    cfg = get_cfg_defaults()
    cfg.model.image_size = 64

    input_tensor = torch.Tensor(np.random.rand(1, 3, 64, 64))

    @pytest.mark.ipus(1)
    @pytest.mark.skip(reason="to enable when loss is implemented")
    def test_training(self):
        model = Yolov4P5(self.cfg)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=False)

        model = trainingModel(model.half(), optimizer=optimizer)
        loss = model(self.input_tensor)
        # TODO implement test when loss is implemented.

    @pytest.mark.ipus(1)
    def test_inference(self):
        self.cfg.inference.nms = False
        model = Yolov4P5(self.cfg)
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
    def test_nms(self):
        cfg = get_cfg()
        transformed_images, transformed_labels, image_sizes = get_image_and_label(cfg)
        model = prepare_model(cfg)
        y = model(transformed_images)
        seen, nt, mean_precision, mean_recall, m_ap50, m_ap = post_process_and_eval(cfg, y, image_sizes, transformed_labels)

        assert seen == 1
        assert nt == 10
        assert mean_precision == 0.4304926702679917
        assert mean_recall == 0.75
        assert m_ap50 == 0.6837500000000001
        assert m_ap == 0.5860250329380764

    @pytest.mark.ipus(1)
    def test_inference_cpu_and_ipu(self):
        self.cfg.model.half = False
        self.cfg.model.image_size = 416
        self.cfg.inference.nms = False

        # Create CPU model
        torch.manual_seed(0)
        self.cfg.model.ipu = False
        model = Yolov4P5(self.cfg)
        cpu_model = model.eval()
        y_cpu = cpu_model(self.input_tensor)

        # Create IPU model
        torch.manual_seed(0)
        self.cfg.model.ipu = True
        model = Yolov4P5(self.cfg)
        ipu_model = inferenceModel(model.eval())
        y_ipu = ipu_model(self.input_tensor)

        assert torch.max(torch.abs(y_cpu[0] - y_ipu[0])) <= 0.002
        assert torch.max(torch.abs(y_cpu[1] - y_ipu[1])) <= 0.002
        assert torch.max(torch.abs(y_cpu[2] - y_ipu[2])) <= 0.002

    def test_fused_inference(self):
        self.cfg.model.normalization = 'batch'
        self.cfg.model.half = False
        self.cfg.inference.nms = False

        model = Yolov4P5(self.cfg)
        before_fuse_model = inferenceModel(model.eval())
        before_fuse_output = before_fuse_model(self.input_tensor)

        model.optimize_for_inference()
        after_fuse_model = inferenceModel(model.eval())
        after_fuse_output = after_fuse_model(self.input_tensor)

        assert torch.max(torch.abs(after_fuse_output[0] - before_fuse_output[0])) <= 1e-4
        assert torch.max(torch.abs(after_fuse_output[1] - before_fuse_output[1])) <= 1e-4
        assert torch.max(torch.abs(after_fuse_output[2] - before_fuse_output[2])) <= 1e-4
