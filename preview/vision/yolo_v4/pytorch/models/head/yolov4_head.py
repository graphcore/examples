# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Tuple
import torch.nn as nn
import torch
from utils.anchors import AnchorBoxes
from utils.custom_ops import CopyTensor


class Yolov4Head(nn.Module):
    """Yolo detection head operating on a single feature tensor."""

    def __init__(
        self,
        anchors: AnchorBoxes,
        num_input_channels: int,
        num_classes: int,
        stride: int,
        precision: torch.dtype = torch.float,
        cpu_mode: bool = False,
    ):
        super().__init__()
        self.anchors = anchors
        # For each anchor box, predict objectness score, box center coords, height, width
        self.num_output_channels = len(anchors) * (num_classes + 5)
        self.detect = nn.Conv2d(num_input_channels, self.num_output_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.num_outputs = num_classes + 5
        self.stride = stride
        self._anchor_centers = None
        self.cpu_mode = cpu_mode
        self.copy_tensor = CopyTensor(self.cpu_mode)
        self.precision = precision

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if self.precision == torch.float:
            feature = feature.to(dtype=self.precision)
        x = self.detect(feature)
        micro_batch_size, _, num_rows, num_cols = x.shape
        x = x.reshape(micro_batch_size, len(self.anchors), self.num_outputs, num_rows, num_cols)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return x

        preds = self.sigmoid(x)

        point_xy_weight_height_class_conf = self.copy_tensor(preds[..., 0:5])[0]
        point_xy = point_xy_weight_height_class_conf[..., 0:2]
        weight_height = point_xy_weight_height_class_conf[..., 2:4]
        class_conf = point_xy_weight_height_class_conf[..., 4].unsqueeze(axis=-1)

        # Predicted box centers are shifted, scaled and added to the anchor_centers, then corrected for stride.
        # TODO remove the dtype=self.precision when constant are auto casted
        xy_coord = (
            self.anchor_centers(num_rows, num_cols)
            + point_xy * torch.tensor([2.0], dtype=self.precision)
            - torch.tensor([0.5], dtype=self.precision)
        ) * torch.tensor([self.stride], dtype=self.precision)

        # Predicted box width and height
        width, height = torch.split((weight_height * torch.tensor([2.0], dtype=self.precision)), [1, 1], dim=4)
        width = (width * width) * self.anchors.widths.reshape(1, -1, 1, 1, 1)
        height = (height * height) * self.anchors.heights.reshape(1, -1, 1, 1, 1)

        final_preds = torch.cat([xy_coord, width, height, class_conf, preds[..., 5:]], dim=-1)

        return final_preds.view(micro_batch_size, -1, self.num_outputs)

    def anchor_centers(self, num_rows: int, num_cols: int) -> torch.Tensor:
        center_y, center_x = torch.meshgrid([torch.arange(num_rows), torch.arange(num_cols)])
        anchor_centers = torch.stack((center_x, center_y), 2)
        return anchor_centers.expand(1, 1, num_rows, num_cols, 2)
