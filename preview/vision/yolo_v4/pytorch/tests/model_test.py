# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Inference test that runs a forward pass and compares results with a GPU reference.
import pytest

import torch
import poptorch

from utils.postprocessing import IPUPredictionsPostProcessing
from tests.test_tools import get_image_and_label, prepare_model, get_cfg, post_process_and_eval


class TestYolov4P5:
    """Tests for end-to-end training and infernece of Yolov4P5."""

    cfg = get_cfg()
    transformed_images, transformed_labels, image_sizes = get_image_and_label(cfg)

    @pytest.mark.ipus(1)
    @pytest.mark.skip(reason="To be enabled when loss is implemented")
    def test_training(self):
        model = prepare_model(self.cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=False)

        model = trainingModel(model.half(), optimizer=optimizer)
        loss = model(self.transformed_images)
        # TODO implement test when loss is implemented.

    @pytest.mark.ipus(1)
    @pytest.mark.skip(reason="To be enabled for debugging purposes only")
    def test_model_and_nms_standalone(self):
        cfg = get_cfg()
        cfg_inference = cfg.inference
        cfg.inference.nms = False

        # Create IPU model
        just_fwd_pass_ipu = prepare_model(cfg)
        predictions_just_fwd_ipu = just_fwd_pass_ipu(self.transformed_images)

        # Create CPU model
        cfg.model.ipu = False
        just_fwd_pass_cpu = prepare_model(cfg)
        predictions_just_fwd_cpu = just_fwd_pass_cpu(self.transformed_images)

        just_nms_ipu = poptorch.inferenceModel(IPUPredictionsPostProcessing(cfg_inference, cpu_mode=False))
        just_nms_cpu = IPUPredictionsPostProcessing(cfg_inference, cpu_mode=True)

        nms_stdaln_ipu = just_nms_ipu(torch.cat(predictions_just_fwd_ipu, axis=1))
        nms_stdaln_cpu = just_nms_cpu(torch.cat(predictions_just_fwd_cpu, axis=1))

        seen, nt, m_precision, m_recall, m_ap50, m_ap = post_process_and_eval(
            cfg, nms_stdaln_ipu, self.image_sizes, self.transformed_labels
        )
        seen_cpu, nt_cpu, m_precision_cpu, m_recall_cpu, m_ap50_cpu, m_ap_cpu = post_process_and_eval(
            cfg, nms_stdaln_cpu, self.image_sizes, self.transformed_labels
        )

        assert seen == seen_cpu
        assert nt == nt_cpu
        assert abs(m_precision - m_precision_cpu) <= 1e-3
        assert abs(m_recall - m_recall_cpu) <= 1e-3
        assert abs(m_ap50 - m_ap50_cpu) <= 1e-3
        assert abs(m_ap - m_ap_cpu) <= 1e-3

    @pytest.mark.ipus(1)
    @pytest.mark.skip(reason="To be enabled for debugging purposes only")
    def test_model_and_nms_end_to_end(self):
        cfg = get_cfg()
        cfg_inference = cfg.inference
        cfg.inference.nms = True

        # Create IPU model
        fwd_pass_and_nms_ipu = prepare_model(cfg, debugging_nms=True)
        nms_output_ipu, predictions_just_fwd_ipu = fwd_pass_and_nms_ipu(self.transformed_images)

        # Create CPU model
        cfg.model.ipu = False
        fwd_pass_and_nms_cpu = prepare_model(cfg, debugging_nms=True)
        nms_output_cpu, predictions_just_fwd_cpu = fwd_pass_and_nms_cpu(self.transformed_images)

        assert torch.max(torch.abs(predictions_just_fwd_cpu[0] - predictions_just_fwd_ipu[0])) <= 1e-3
        assert torch.max(torch.abs(predictions_just_fwd_cpu[1] - predictions_just_fwd_ipu[1])) <= 1e-3
        assert torch.max(torch.abs(predictions_just_fwd_cpu[2] - predictions_just_fwd_ipu[2])) <= 1e-3

        just_nms_ipu = poptorch.inferenceModel(PredictionsPostProcessing(cfg_inference, cpu_mode=False))
        just_nms_cpu = PredictionsPostProcessing(cfg_inference, cpu_mode=True)

        nms_stdaln_ipu = just_nms_ipu(torch.cat(predictions_just_fwd_ipu, axis=1))
        nms_stdaln_cpu = just_nms_cpu(torch.cat(predictions_just_fwd_cpu, axis=1))

        seen_stdaln, nt_stdaln, m_stdaln_precision, m_stdaln_recall, m_stdaln_ap50, m_stdaln_ap = post_process_and_eval(
            cfg, nms_stdaln_ipu, self.image_sizes, self.transformed_labels
        )
        seen, nt, m_precision, m_recall, m_ap50, m_ap = post_process_and_eval(
            cfg, nms_output_ipu, self.image_sizes, self.transformed_labels
        )
        seen_cpu, nt_cpu, m_precision_cpu, m_recall_cpu, m_ap50_cpu, m_ap_cpu = post_process_and_eval(
            cfg, nms_stdaln_cpu, self.image_sizes, self.transformed_labels
        )

        assert seen_stdaln == seen_cpu
        assert nt_stdaln == nt_cpu
        assert abs(m_stdaln_precision - m_precision_cpu) <= 1e-3
        assert abs(m_stdaln_recall - m_recall_cpu) <= 1e-3
        assert abs(m_stdaln_ap50 - m_ap50_cpu) <= 1e-3
        assert abs(m_stdaln_ap - m_ap_cpu) <= 1e-3

        assert seen_stdaln == seen
        assert nt_stdaln == nt
        assert abs(m_stdaln_precision - m_precision) <= 1e-3
        assert abs(m_stdaln_recall - m_recall) <= 1e-3
        assert abs(m_stdaln_ap50 - m_ap50) <= 1e-3
        assert abs(m_stdaln_ap - m_ap) <= 1e-3

        assert seen == seen_cpu
        assert nt == nt_cpu
        assert abs(m_precision - m_precision_cpu) <= 1e-3
        assert abs(m_recall - m_recall_cpu) <= 1e-3
        assert abs(m_ap50 - m_ap50_cpu) <= 1e-3
        assert abs(m_ap - m_ap_cpu) <= 1e-3

    @pytest.mark.ipus(1)
    def test_inference_and_nms(self):
        cfg = get_cfg()
        ipu_model = prepare_model(cfg)
        y_ipu = ipu_model(self.transformed_images)

        seen, nt, m_precision, m_recall, m_ap50, m_ap = post_process_and_eval(
            cfg, y_ipu, self.image_sizes, self.transformed_labels
        )

        cfg.model.ipu = False
        model = prepare_model(cfg)
        y_cpu = model(self.transformed_images)
        seen_cpu, nt_cpu, m_precision_cpu, m_recall_cpu, m_ap50_cpu, m_ap_cpu = post_process_and_eval(
            cfg, y_cpu, self.image_sizes, self.transformed_labels
        )

        assert seen == seen_cpu
        assert nt == nt_cpu
        assert abs(m_precision - m_precision_cpu) <= 1e-3
        assert abs(m_recall - m_recall_cpu) <= 1e-3
        assert abs(m_ap50 - m_ap50_cpu) <= 1e-3
        assert abs(m_ap - m_ap_cpu) <= 1e-3

    @pytest.mark.ipus(1)
    @pytest.mark.skip(reason="To be enabled for debugging purposes only")
    def test_inference_cpu_and_ipu(self):
        cfg = get_cfg()
        cfg.inference.nms = False

        # Create IPU model
        ipu_model = prepare_model(cfg)
        y_ipu = ipu_model(self.transformed_images)

        # Create CPU model
        cfg.model.ipu = False
        model = prepare_model(cfg)
        y_cpu = model(self.transformed_images)

        expected_output_size = model.output_shape((cfg.model.image_size, cfg.model.image_size))

        p3 = expected_output_size["p3"]
        p4 = expected_output_size["p4"]
        p5 = expected_output_size["p5"]

        assert y_ipu[0].shape == torch.Size([p3[0], p3[1] * p3[2] * p3[3], p3[4]])
        assert y_ipu[1].shape == torch.Size([p4[0], p4[1] * p4[2] * p4[3], p4[4]])
        assert y_ipu[2].shape == torch.Size([p5[0], p5[1] * p5[2] * p5[3], p5[4]])
        assert torch.max(torch.abs(y_cpu[0] - y_ipu[0])) <= 1e-3
        assert torch.max(torch.abs(y_cpu[1] - y_ipu[1])) <= 1e-3
        assert torch.max(torch.abs(y_cpu[2] - y_ipu[2])) <= 1e-3
