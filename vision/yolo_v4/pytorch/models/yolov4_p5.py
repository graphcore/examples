# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, Tuple
from yacs.config import CfgNode

import torch
import torch.nn as nn

from models.backbone.yolov4_p5 import Yolov4P5BackBone
from models.detector import Detector
from models.head.yolov4_head import Yolov4Head
from models.layers import Mish
from models.loss import Yolov4_loss
from models.neck.yolov4_p5 import Yolov4P5Neck
from utils.anchors import AnchorBoxes
from utils.postprocessing import IPUPredictionsPostProcessing


class Yolov4P5(Detector):
    """
    Yolov4P5 object detector as described in https://arxiv.org/abs/2011.08036.
    """

    def __init__(
        self,
        cfg: CfgNode,
        backbone: nn.Module = Yolov4P5BackBone,
        neck: nn.Module = Yolov4P5Neck,
        detector_head: nn.Module = Yolov4Head,
        debugging_nms: bool = False,
    ):
        super().__init__(backbone, neck, detector_head)
        self.cpu_mode = not cfg.model.ipu
        self.debugging_nms = debugging_nms

        # We storage the specific parameters of training or inference,
        # for example for inference we have nms, and it's hyperparameters.
        if cfg.model.mode == "train":
            self.nms = False
            self.loss = Yolov4_loss(cfg)
            self.training = True
        else:
            specific_mode_parameters = cfg.inference
            self.nms = specific_mode_parameters.nms
            self.ipu_post_process = IPUPredictionsPostProcessing(specific_mode_parameters, self.cpu_mode)
            self.training = False

        cfg = cfg.model

        if cfg.activation == "relu":
            activation = nn.ReLU()
        elif cfg.activation == "mish":
            activation = Mish()
        else:
            activation = nn.Linear()

        self.precision = torch.float16 if cfg.precision == "half" else torch.float32

        self.anchors = {
            "p3": AnchorBoxes(
                widths=torch.tensor(cfg.anchors.p3width, requires_grad=False),
                heights=torch.tensor(cfg.anchors.p3height, requires_grad=False),
            ),
            "p4": AnchorBoxes(
                widths=torch.tensor(cfg.anchors.p4width, requires_grad=False),
                heights=torch.tensor(cfg.anchors.p4height, requires_grad=False),
            ),
            "p5": AnchorBoxes(
                widths=torch.tensor(cfg.anchors.p5width, requires_grad=False),
                heights=torch.tensor(cfg.anchors.p5height, requires_grad=False),
            ),
        }

        self.micro_batch_size = cfg.micro_batch_size
        self.n_classes = cfg.n_classes
        self.strides = cfg.strides

        self.uint_io = cfg.uint_io

        if self.cpu_mode or cfg.precision != "half":
            self.model_dtype = "float"
        else:
            self.model_dtype = "half"

        self.backbone = backbone(cfg.input_channels, activation, cfg.normalization)
        self.neck = neck(activation, cfg.normalization)

        # TODO remove precision when constant are auto-casted
        self.headp3 = detector_head(
            self.anchors["p3"], 256, self.n_classes, self.strides[0], precision=self.precision, cpu_mode=self.cpu_mode
        )
        self.headp4 = detector_head(
            self.anchors["p4"], 512, self.n_classes, self.strides[1], precision=self.precision, cpu_mode=self.cpu_mode
        )
        self.headp5 = detector_head(
            self.anchors["p5"], 1024, self.n_classes, self.strides[2], precision=self.precision, cpu_mode=self.cpu_mode
        )

    def change_input_type(self, x: torch.Tensor) -> torch.Tensor:
        if self.uint_io:
            if self.model_dtype == "float":
                x = x.float() / 255.0
            else:
                x = x.half() / 255.0
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        x = self.change_input_type(x)

        x = self.backbone(x)
        p5, p4, p3 = self.neck(x)
        p3 = self.headp3(p3)
        p4 = self.headp4(p4)
        p5 = self.headp5(p5)

        predictions = (p3, p4, p5)

        if self.nms:
            if self.debugging_nms:
                return self.ipu_post_process(torch.cat(predictions, axis=1)), predictions
            else:
                return self.ipu_post_process(torch.cat(predictions, axis=1))

        if self.training:
            return predictions, self.loss(predictions, y)

        return predictions

    def output_shape(self, input_shape: Tuple[int, int]) -> Dict[str, Tuple[int, ...]]:
        if len(input_shape) != 2:
            raise ValueError("`input_shape` must be tuple of length 2 (img_width, img_height).")
        p3_size = [int(i / self.strides[0]) for i in input_shape]
        p4_size = [int(i / self.strides[1]) for i in input_shape]
        p5_size = [int(i / self.strides[2]) for i in input_shape]
        return {
            "p3": [self.micro_batch_size, len(self.anchors["p3"]), *p3_size, self.n_classes + 5],
            "p4": [self.micro_batch_size, len(self.anchors["p4"]), *p4_size, self.n_classes + 5],
            "p5": [self.micro_batch_size, len(self.anchors["p5"]), *p5_size, self.n_classes + 5],
        }
