# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Dict, Tuple
from yacs.config import CfgNode

import torch
import torch.nn as nn

from models.backbone.yolov4_p5 import Yolov4P5BackBone
from models.detector import Detector
from models.head.yolov4_head import Yolov4Head
from models.neck.yolov4_p5 import Yolov4P5Neck
from models.layers import Mish
from utils.anchors import AnchorBoxes
from utils.postprocessing import PredictionsPostProcessing


class PreprocessTargets(nn.Module):
    """
    Implementation of IPU label preprocessing for yolov4 loss
    """
    def __init__(self, cfg, tmp_anchors=None):
        super().__init__()
        self.strides = torch.tensor(cfg.strides)
        self.num_classes = cfg.n_classes
        self.image_size = cfg.image_size
        self.input_channels = cfg.input_channels
        self.train_input_sizes = [int(self.image_size/stride) for stride in self.strides]

        self.precision = (torch.float16 if cfg.half else torch.float32)

        if tmp_anchors is None:
            tmp_anchors = [AnchorBoxes(widths=torch.tensor(cfg.anchors.p3width, dtype=self.precision),
                                       heights=torch.tensor(cfg.anchors.p3height, dtype=self.precision)),
                           AnchorBoxes(widths=torch.tensor(cfg.anchors.p4width, dtype=self.precision),
                                       heights=torch.tensor(cfg.anchors.p4height, dtype=self.precision)),
                           AnchorBoxes(widths=torch.tensor(cfg.anchors.p5width, dtype=self.precision),
                                       heights=torch.tensor(cfg.anchors.p5height, dtype=self.precision))]

        self.auto_anchors = cfg.auto_anchors
        self.anchor_per_scale = len(tmp_anchors[0])

        self.max_n_labels = [int(cfg.max_nlabels_p3), int(cfg.max_nlabels_p4), int(cfg.max_nlabels_p5)]

        self.anchors = [(tmp_anchor.to_torch_tensor().view(2, self.anchor_per_scale).transpose(1, 0).to(self.precision).contiguous() / self.strides[i]) for i, tmp_anchor in enumerate(tmp_anchors)]

        self.anchor_threshold = cfg.anchor_threshold

    def get_indices_tbox(self, labels, idx_detection_head):
        # Input: Labels[batch_size, max_n_labels, 5] class,x,y,w,h
        # Output: anchor_ind [5, batch_size, max_n_labels, n_anchors_per_scale, 1] indices for anchors
        #         yind [5, batch_size, max_n_labels, n_anchors_per_scale, 1] indices for y
        #         xind [5, batch_size, max_n_labels, n_anchors_per_scale, 1] indices for x
        #         t_boxes [5, batch_size, max_n_labels, n_anchors_per_scale, 5] true values for the boxes
        b_size = labels.shape[0]
        n_labels = labels.shape[1]
        g = 0.5
        off = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]]) * g

        feature_map_size = self.train_input_sizes[idx_detection_head]

        classes = labels[:, :, 0].unsqueeze(axis=-1)
        xywh = labels[:, :, 1:]
        xywh = xywh * feature_map_size
        labels = torch.cat((classes, xywh), axis=-1)

        labels = labels.unsqueeze(axis=2).repeat(1, 1, self.anchor_per_scale, 1)
        anchor_ind = torch.arange(1., float(self.anchor_per_scale)+1.).view(1, 1, self.anchor_per_scale, 1).repeat(b_size, 1, n_labels, 1).view(b_size, n_labels, self.anchor_per_scale, 1)
        labels = torch.cat((labels, anchor_ind), axis=-1)

        # We check the difference between the ratio of all the anchors with all the labels
        # If some are under a threshold we choose them as good enough
        labels_wh = labels[:, :, :, 3:5]
        ratio_anchors = labels_wh / self.anchors[idx_detection_head].unsqueeze(axis=0)
        # labels_wh
        # tensor([[[ 2.,  2.],
        #          [ 1.,  3.],
        #          [ 1.,  1.],
        #          [10.,  5.]]])
        # anchors
        # tensor([[[2., 2.],
        #         [2., 2.],
        #         [2., 2.],
        #         [2., 2.]]])
        # ratio_a = labels_wh / anchors
        # If the ratios are bigger than the anchor then ratios will be > 1.
        # if the ratios are smaller then the anchor then 1/ratios will be > 1.
        # ratio_a
        # tensor([[[1.0000, 1.0000],
        #          [0.5000, 1.5000],
        #          [0.5000, 0.5000],
        #          [5.0000, 2.5000]]])
        # 1.0/ratio_a
        # tensor([[[1.0000, 1.0000],
        #          [2.0000, 0.6667],
        #          [2.0000, 2.0000],
        #          [0.2000, 0.4000]]])
        # We calculate the worse ratio in each dimension
        worse_ratios_wh = torch.max(ratio_anchors, 1./ratio_anchors)
        # Then we get the worst ratio for each label per anchor in any of the 2 dimensions
        # and we compare to a hyper parameter if the worst case is smaller then is a good fit
        mask = torch.amax(worse_ratios_wh, 3)[0]
        mask = torch.where(mask != 0., mask, torch.tensor([torch.finfo(torch.half).max]))
        mask = torch.lt(mask, torch.tensor([self.anchor_threshold]))

        # We mask the labels and the anchor indexes, these are the best labels for our anchors
        best_targets = torch.where(mask.unsqueeze(axis=-1).repeat(1, 1, 1, labels.shape[-1]), labels, torch.tensor([0.]))

        # We get the point x and y
        best_xy = best_targets[:, :, :, 1:3]
        # Do the inverse of the point:
        best_inverse_xy = torch.where(best_xy != 0., feature_map_size - best_xy, torch.tensor([0.]))

        # We mask as true all the pixels that the decimal part is smaller than 0.5 and not located in the right up border.
        # For example, 3.2 .2 is smaller than g=0.5 and 3. is greater than 1.
        x_mask, y_mask = ((best_xy % 1. < g) * (best_xy > 1.)).permute(3, 0, 1, 2)

        # We mask as true all the pixels for the inverse x and y that the decimal part is smaller than 0.5 and not located in the inverse right up border.
        # For example, if the original x is 3.2 and the image size is 10 then the inverse of 3.2 is "10 - 3.2 = 6.8" which is False for
        # the first comparison since .8 is greater than g=0.5 but True for the second one since 6. is greater than 1.
        # However, how we are doing a "logical and" between the comparisons "False & True = False"
        inverse_x_mask, inverse_y_mask = ((best_inverse_xy % 1. < g) * (best_inverse_xy > 1.)).permute(3, 0, 1, 2)

        # Now we have 5 masks the first one full of Trues which will get all the labels and 4 extra masks with the previous mentioned criteria
        indices = torch.stack((torch.ones_like(x_mask, dtype=int).bool(), x_mask, y_mask, inverse_x_mask, inverse_y_mask))

        # Now we mask with those combinations
        targets = torch.where(indices.unsqueeze(axis=-1).repeat(1, 1, 1, 1, best_targets.shape[-1]), best_targets.unsqueeze(axis=0).repeat(indices.shape[0], 1, 1, 1, 1), torch.tensor([0.]))

        # Offsets will decrease or increase the values for the masks where we checked if the decimal part was greater or smaller than 0.5
        # For all the x, y values smaller than 0.5 we will add 0.5 for all the values grester than 0.5 we will add -0.5
        offsets = torch.where(indices.unsqueeze(axis=-1).repeat(1, 1, 1, 1, off.shape[-1]), off.view(indices.shape[0], 1, 1, 1, off.shape[-1]).repeat(1, b_size, n_labels, self.anchor_per_scale, 1), torch.tensor([0.]))

        # We flatten the tensors to index them efficiently
        targets = targets.view(-1, targets.shape[-1])
        offsets = offsets.view(-1, offsets.shape[-1])

        # We calculate the the addition of all the elements
        # to get the non-zero elements, we sum all the dimension
        # because some classes might be 0. and the same for x, y and anchor index
        cxywh_added = torch.sum(targets, dim=-1)

        # This return is used when we want to get the maximum number of labels
        # after preprocessing
        if self.auto_anchors:
            return torch.sum(cxywh_added.bool())

        # We have lots of padding in the mask so we reduce some by taking the maximum
        # number of preprocessing images, by default is the worst case scenario
        # but it can be refined by sweeping through the dataset per image size
        _, idx_filter = torch.topk(cxywh_added, self.max_n_labels[idx_detection_head])
        idx_filter = idx_filter.float().long()

        # We use the indexes to eliminate some of the padding
        targets = torch.index_select(input=targets, dim=0, index=idx_filter)
        offsets = torch.index_select(input=offsets, dim=0, index=idx_filter)

        # We get the different elements from the targets
        classes = targets[:, 0].unsqueeze(axis=-1)
        label_xy = targets[:, 1:3]
        label_wh = targets[:, 3:5]

        # We calculate the indices of x and y
        label_xy_indices = torch.where(label_xy != 0., (label_xy - offsets).long(), torch.tensor([0.], dtype=torch.long))

        # Make sure that x, y, and anchors stay within their range
        x_ind, y_ind = label_xy_indices.permute(1, 0).clamp_(0, feature_map_size-1).long()
        anchor_ind = (targets[:, 5] - 1).clamp_(0, 4).long()

        # Finally, we create the t_boxes with class, x_offset, y_offset, width and height
        t_boxes = torch.cat((classes, (label_xy - label_xy_indices), label_wh), axis=-1)

        return anchor_ind, y_ind, x_ind, t_boxes

    def forward(self, real_labels):
        t_indices_boxes_p5 = None
        t_indices_boxes_p4 = None
        t_indices_boxes_p3 = None

        t_indices_boxes_p5 = self.get_indices_tbox(real_labels, idx_detection_head = 0)

        t_indices_boxes_p4 = self.get_indices_tbox(real_labels, idx_detection_head = 1)

        t_indices_boxes_p3 = self.get_indices_tbox(real_labels, idx_detection_head = 2)

        return t_indices_boxes_p5, t_indices_boxes_p4, t_indices_boxes_p3


class Yolov4P5(Detector):
    """
    Yolov4P5 object detector as described in https://arxiv.org/abs/2011.08036.
    """

    def __init__(self, cfg: CfgNode, backbone: nn.Module = Yolov4P5BackBone, neck: nn.Module = Yolov4P5Neck, detector_head: nn.Module = Yolov4Head):
        super().__init__(backbone, neck, detector_head)
        self.cpu_mode = not cfg.model.ipu

        # We storage the specific paremeters of training or inference,
        # for example for inference we have nms, and it's hyperparameters.
        if cfg.model.mode == "train":
            specific_mode_parameters = cfg.training
            self.nms = False
        else:
            specific_mode_parameters = cfg.inference
            self.nms = specific_mode_parameters.nms
            self.ipu_post_process = PredictionsPostProcessing(specific_mode_parameters, self.cpu_mode)

        cfg = cfg.model

        if cfg.activation == "relu":
            activation = nn.ReLU()
        elif cfg.activation == "mish":
            activation = Mish()
        else:
            activation = nn.Linear()

        self.precision = (torch.float16 if cfg.half else torch.float32)

        self.anchors = {"p3": AnchorBoxes(widths=torch.tensor(cfg.anchors.p3width, requires_grad=False),
                                          heights=torch.tensor(cfg.anchors.p3height, requires_grad=False)),
                        "p4": AnchorBoxes(widths=torch.tensor(cfg.anchors.p4width, requires_grad=False),
                                          heights=torch.tensor(cfg.anchors.p4height, requires_grad=False)),
                        "p5": AnchorBoxes(widths=torch.tensor(cfg.anchors.p5width, requires_grad=False),
                                          heights=torch.tensor(cfg.anchors.p5height, requires_grad=False))}

        self.micro_batch_size = cfg.micro_batch_size
        self.n_classes = cfg.n_classes
        self.strides = cfg.strides

        self.uint_io = cfg.uint_io

        if self.cpu_mode or not cfg.half:
            self.model_dtype = "float"
        else:
            self.model_dtype = "half"

        self.backbone = backbone(
            cfg.input_channels, activation, cfg.normalization)
        self.neck = neck(activation, cfg.normalization)

        # TODO remove precision when constant are auto-casted
        self.headp3 = detector_head(self.anchors["p3"], 256, self.n_classes, self.strides[0], precision=self.precision, cpu_mode=self.cpu_mode)
        self.headp4 = detector_head(self.anchors["p4"], 512, self.n_classes, self.strides[1], precision=self.precision, cpu_mode=self.cpu_mode)
        self.headp5 = detector_head(self.anchors["p5"], 1024, self.n_classes, self.strides[2], precision=self.precision, cpu_mode=self.cpu_mode)

    def change_input_type(self, x: torch.Tensor) -> torch.Tensor:
        if self.uint_io:
            if self.model_dtype == "float":
                x = x.float() / 255.
            else:
                x = x.half() / 255.
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
            return self.ipu_post_process(torch.cat(predictions, axis=1))
        else:
            return predictions

    def output_shape(self, input_shape: Tuple[int, int]) -> Dict[str, Tuple[int, ...]]:
        if len(input_shape) != 2:
            raise ValueError(
                "`input_shape` must be tuple of length 2 (img_width, img_height).")
        p3_size = [int(i / self.strides[0]) for i in input_shape]
        p4_size = [int(i / self.strides[1]) for i in input_shape]
        p5_size = [int(i / self.strides[2]) for i in input_shape]
        return {"p3": [self.micro_batch_size, len(self.anchors["p3"]), *p3_size, self.n_classes + 5],
                "p4": [self.micro_batch_size, len(self.anchors["p4"]), *p4_size, self.n_classes + 5],
                "p5": [self.micro_batch_size, len(self.anchors["p5"]), *p5_size, self.n_classes + 5]}
