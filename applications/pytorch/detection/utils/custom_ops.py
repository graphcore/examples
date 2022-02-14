# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import ctypes
import os
from pathlib import Path
from typing import List
from yacs.config import CfgNode

import torch

import poptorch

from utils.tools import nms as cpu_nms


def load_custom_ops_lib(path_custom_op: str):
    """Loads the custom op binary
        Parameters:
            predictions (str): name of the custom op binary file
    """
    path_to_detection = Path(os.environ['PYTORCH_APPS_DETECTION_PATH'])
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

    def forward(self, scores: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor = None) -> List[torch.Tensor]:
        batch = scores.shape[0]
        if self.cpu_mode:
            cpu_output = cpu_nms(scores, boxes, classes, self.iou_threshold, self.nms_max_detections)
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
