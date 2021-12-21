# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import ctypes
import os
from pathlib import Path
from typing import List
from yacs.config import CfgNode

import torch
from torchvision.ops.boxes import nms as torchvision_nms

import poptorch


def load_custom_ops_lib(path_custom_op: str):
    """Loads the custom op binary
        Parameters:
            predictions (str): name of the custom op binary file
    """
    path_to_detection = Path(__file__).parent.parent
    os.environ['PYTORCH_APPS_DETECTION_PATH'] = str(path_to_detection)
    so_path = path_to_detection.joinpath("utils/custom_ops/build/" + path_custom_op)

    if not so_path.is_file():
        print("Build the custom ops library with `make` before running this script")
        print("Couldn't find file", str(so_path))
        exit(1)

    ctypes.cdll.LoadLibrary(str(so_path))


class CopyTensor(torch.nn.Module):
    def __init__(self, cpu_mode: bool):
        super().__init__()
        self.cpu_mode = cpu_mode
        load_custom_ops_lib("copy_tensor_custom_op.so")

    def cpu_copy_tensor(self, input_: torch.Tensor) -> List[torch.Tensor]:
        return [input_]

    def forward(self, input_: torch.Tensor) -> List[torch.Tensor]:
        cpu_output = self.cpu_copy_tensor(input_)

        if self.cpu_mode:
            return cpu_output

        return poptorch.custom_op(
            inputs=[input_],
            name="CopyTensor",
            domain="ai.graphcore",
            domain_version=1,
            example_outputs=cpu_output,
        )


class Nms(torch.nn.Module):
    def __init__(self, inference_cfg: CfgNode, cpu_mode: bool):
        super().__init__()
        load_custom_ops_lib("nms_custom_op.so")
        self.iou_threshold = inference_cfg.iou_threshold
        self.score_threshold = inference_cfg.class_conf_threshold
        self.nms_max_detections = inference_cfg.nms_max_detections
        self.cpu_mode = cpu_mode

    def cpu_nms(self, scores: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor) -> List[torch.Tensor]:
        batch = scores.shape[0]
        selected_box_indx = torch.full((batch, self.nms_max_detections), -1, dtype=torch.long)
        cpu_classes = torch.full((batch, self.nms_max_detections), torch.iinfo(torch.int32).max, dtype=int)
        cpu_boxes = torch.zeros((batch, self.nms_max_detections, 4))
        cpu_scores = torch.zeros((batch, self.nms_max_detections))
        cpu_true_max_detections = torch.full((batch,), -1)

        for i, (bscores, bboxes, bclasses) in enumerate(zip(scores, boxes, classes)):
            nms_preds = torchvision_nms(bboxes, bscores, self.iou_threshold)

            if nms_preds.shape[0] > self.nms_max_detections:
                selected_box_indx[i] = nms_preds[:self.nms_max_detections]
            else:
                selected_box_indx[i, :nms_preds.shape[0]] = nms_preds
                cpu_true_max_detections[i] = nms_preds.shape[0]

            batch_indices = selected_box_indx[i, :cpu_true_max_detections[i]]

            cpu_classes[i, :cpu_true_max_detections[i]] = bclasses[batch_indices]
            cpu_boxes[i, :cpu_true_max_detections[i]] = bboxes[batch_indices]
            cpu_scores[i, :cpu_true_max_detections[i]] = bscores[batch_indices]

        return [selected_box_indx, cpu_scores, cpu_boxes, cpu_classes.int(), cpu_true_max_detections.int()]

    def forward(self, scores: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor = None) -> List[torch.Tensor]:
        batch = scores.shape[0]
        if self.cpu_mode:
            cpu_output = self.cpu_nms(scores, boxes, classes)
            return cpu_output

        return poptorch.custom_op(
            inputs=[scores, boxes],
            name="Nms",
            domain="ai.graphcore",
            domain_version=1,
            attributes={"threshold": self.iou_threshold, "scoreThreshold": self.score_threshold, "numDetections": self.nms_max_detections, "useGather": 1},
            example_outputs=[torch.zeros(dtype=torch.long, size=[batch, self.nms_max_detections]),
                             torch.zeros(dtype=scores.dtype, size=[batch, self.nms_max_detections]),
                             torch.zeros(dtype=boxes.dtype, size=[batch, self.nms_max_detections, 4]),
                             torch.zeros(dtype=torch.int, size=[batch, self.nms_max_detections]),
                             torch.zeros(dtype=torch.int, size=[batch])]
        )
