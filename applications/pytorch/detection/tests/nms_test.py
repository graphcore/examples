# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest

import torch
import poptorch

from utils.custom_ops import Nms
from utils.postprocessing import PredictionsPostProcessing
from tests.test_tools import get_image_and_label, prepare_model, get_cfg, post_process_and_eval


class TestNms:
    """Tests nms custom op"""
    def prepare_test(self):
        cfg = get_cfg()
        cfg.model.ipu = False
        cfg.inference.nms = False
        transformed_images, transformed_labels, image_sizes = get_image_and_label(cfg)
        model = prepare_model(cfg)
        y = model(transformed_images)
        predictions = torch.cat(y, axis=1)
        cpu_preprocessing = PredictionsPostProcessing(cfg.inference, cpu_mode=True).nms_preprocessing(predictions)
        cfg.inference.nms = True
        cfg.model.ipu = True
        return cfg, (predictions, transformed_labels, image_sizes), cpu_preprocessing

    @pytest.mark.ipus(1)
    def test_preprocessing_and_nms_standalone(self):
        cfg, model_input, cpu_preprocessing = self.prepare_test()
        cfg_inference = cfg.inference
        predictions = model_input[0]

        ipu_model = PredictionsPostProcessing(cfg_inference, cpu_mode=False, testing_preprocessing=True)
        cpu_model = PredictionsPostProcessing(cfg_inference, cpu_mode=True, testing_preprocessing=True)

        ipu_model = poptorch.inferenceModel(ipu_model)

        ipu_preprocessing = ipu_model(predictions)
        ipu_preprocessing_on_cpu = cpu_model(predictions)

        scores_ipu = ipu_preprocessing[0]
        scores_cpu = ipu_preprocessing_on_cpu[0]

        assert torch.all((scores_ipu - scores_cpu) < 1e-04)

        ipu_model = Nms(cfg_inference, cpu_mode=False)
        cpu_model = Nms(cfg_inference, cpu_mode=True)

        ipu_model = poptorch.inferenceModel(ipu_model)

        scores_ipu = ipu_preprocessing[0]
        boxes_ipu = ipu_preprocessing[1]

        ipu_nms_output = ipu_model(scores_ipu, boxes_ipu)

        scores_cpu = cpu_preprocessing[0]
        boxes_cpu = cpu_preprocessing[1]
        classes_cpu = cpu_preprocessing[2]

        cpu_nms_output = cpu_model(scores_cpu, boxes_cpu, classes_cpu)
        shifting = 4096.
        box_shift = (cpu_nms_output[3].float() * shifting).unsqueeze(axis=-1).float()

        assert torch.all((ipu_nms_output[1] - cpu_nms_output[1]) < 1e-04)
        assert torch.all(torch.abs(ipu_nms_output[2] - (cpu_nms_output[2] - box_shift)) < 5e-03)
        assert torch.equal(ipu_nms_output[3], cpu_nms_output[3])
        assert torch.equal(ipu_nms_output[4], cpu_nms_output[4])

    @pytest.mark.ipus(1)
    def test_end_to_end_nms(self):
        cfg, model_input, cpu_preprocessing = self.prepare_test()
        cfg_inference = cfg.inference
        predictions, transformed_labels, image_sizes = model_input

        ipu_model = poptorch.inferenceModel(PredictionsPostProcessing(cfg_inference, cpu_mode=False))
        cpu_model = PredictionsPostProcessing(cfg_inference, cpu_mode=True)

        result_nms_ipu = ipu_model(predictions)
        result_nms_cpu = cpu_model(predictions)

        predictions_ipu = result_nms_ipu[0]
        predictions_cpu = result_nms_cpu[0]
        max_detections_ipu = result_nms_ipu[1]
        max_detections_cpu = result_nms_cpu[1]

        for i, (prediction_ipu, prediction_cpu) in enumerate(zip(predictions_ipu, predictions_cpu)):
            true_detections_ipu = prediction_ipu[:max_detections_ipu[i]]
            true_detections_cpu = prediction_cpu[:max_detections_cpu[i]]

            scores_ipu = true_detections_ipu[:, 4]
            scores_cpu = true_detections_cpu[:, 4]

            assert torch.all((scores_ipu - scores_cpu) < 1e-04)

        seen, nt, m_precision, m_recall, m_ap50, m_ap = post_process_and_eval(cfg, result_nms_ipu, image_sizes, transformed_labels)
        seen_cpu, nt_cpu, m_precision_cpu, m_recall_cpu, m_ap50_cpu, m_ap_cpu = post_process_and_eval(cfg, result_nms_cpu, image_sizes, transformed_labels)

        assert seen == seen_cpu
        assert nt == nt_cpu
        assert abs(m_precision - m_precision_cpu) <= 1e-3
        assert abs(m_recall - m_recall_cpu) <= 1e-3
        assert abs(m_ap50 - m_ap50_cpu) <= 1e-3
        assert abs(m_ap - m_ap_cpu) <= 1e-3
