# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import List, Tuple

import torch
import numpy as np
from PIL import Image

from utils.tools import ioa, standardize_labels, xywh_to_xyxy
"""
Torch transformations for the YOLOv4 dataset
"""


class Pad(object):
    """
    Pads the image to [self.image_size, self.image_size] and transforms the label to match the padding.
    Call function:
        Parameters:
            item:
                img: image to apply the padding
                labels: labels to apply the transformation
        Return:
            img: image padded to self.image_size
            labels: label transformed to match the same bounding box position after the padding
    """
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, item: Tuple[np.array, np.array]) -> Tuple[np.array, np.array]:
        input_image, labels = item

        output_image = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)

        pad_width = int(np.round(abs(self.image_size-input_image.shape[1])/2))
        pad_height = int(np.round(abs(self.image_size-input_image.shape[0])/2))

        output_image[pad_height:(input_image.shape[0]+pad_height), pad_width:(input_image.shape[1] + pad_width), :] = input_image

        labels[:, [1, 3]] *= input_image.shape[1]/self.image_size
        labels[:, [2, 4]] *= input_image.shape[0]/self.image_size
        labels[:, 1] += (pad_width)/self.image_size
        labels[:, 2] += (pad_height)/self.image_size

        return output_image, labels


class ResizetoNumpy(object):
    """
    Returns a resize image from
    [W_o, H_o] to [self.image_size/max(W_o, H_o)*W_o, self.image_size/max(W_o, H_o)*H_o].
    Call function:
        Parameters:
            item:
                image: image to apply the resize and the cast to numpy
                labels: labels doesn't perform any transformation
        Return:
            image: resized image
            labels: same labels
    """
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[np.array, np.array]:
        image, labels = item

        height, width = image.size

        ratio_image_size = self.image_size / max(height, width)
        if ratio_image_size != 1:
            interp = Image.BILINEAR if ratio_image_size > 1 else Image.NEAREST
            return np.ascontiguousarray(image.resize((int(height * ratio_image_size), int(width * ratio_image_size)), interp)), labels
        else:
            return np.ascontiguousarray(image), labels


class ToTensor(object):
    """
    Convert item (image: np.array[uint], label: np.array float)
    to (image: torch.Tensor[float32], label: torch.Tensor[float32]).
    Call function:
        Parameters:
            item:
                image: image to apply transformation from channel last to channels first,
                    uint to float32 and division by 255.
                labels: labels to pad up to max_bbox_per_scale
        Return:
            image: transformed image
            labels: padded labels
    """
    def __init__(self, max_bbox_per_scale: int, image_type: str):
        self.max_bbox_per_scale = max_bbox_per_scale
        self.image_type = image_type

    def __call__(self, item: Tuple[np.array, np.array]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, input_labels = item

        output_labels = torch.zeros((self.max_bbox_per_scale, input_labels.shape[1]))
        output_labels[:input_labels.shape[0]] = torch.from_numpy(input_labels)

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.image_type == "half":
            image = image.half() / 255.
        elif self.image_type == "float":
            image = image.float() / 255.
        return image, output_labels


# TODO: Add to a transform pipeline once we have train.py
class Cutout(object):
    """
    Apply random patches on the image with random colors to improve
    regularization during training.
    Call function:
        Parameters:
            item:
                image: image to apply the cutout
                labels: labels where the obsecured area will be removed from
        Return:
            image: image with patches applied
            labels: labels with obscured area > obscured_pct removed
    """
    def __init__(self, patch_scales: List[float], obscured_pct: float, scaled_threshold: float = 0.3):
        self.patch_scales = patch_scales
        self.obscured_pct = obscured_pct
        self.scaled_treshold = scaled_threshold

    def __call__(self, item: Tuple[np.array, np.array]) -> Tuple[np.array, np.array]:
        image, labels = item

        height, width = image.shape[0], image.shape[1]

        for scale in self.patch_scales:
            scaled_width = np.random.randint(1, int(width * scale))
            scaled_height = np.random.randint(1, int(height * scale))

            x = np.random.randint(0, width)
            y = np.random.randint(0, height)

            xmin = max(x - scaled_width // 2, 0)
            ymin = max(y - scaled_height // 2, 0)
            xmax = min(width, xmin + scaled_width)
            ymax = min(height, ymin + scaled_height)

            image[ymin:ymax, xmin:xmax] = [np.random.randint(64, 191) for _ in range(3)]

            # translate label from ratio to absolute values and
            # from centerx, centery, width, height to xmin, ymin, xmax, ymax
            scaled_label = labels.copy()
            scaled_label = standardize_labels(scaled_label, width, height)
            scaled_label = xywh_to_xyxy(scaled_label[..., 1:])

            # remove obscured area from the labels
            if len(labels) and scale > self.scaled_treshold:
                cutout_box = np.array([[xmin, ymin, xmax, ymax]])
                pct_intersect = ioa(cutout_box, scaled_label)
                labels = labels[pct_intersect[0] < self.obscured_pct]

        return image, labels


class HSV(object):
    """
    Apply HSV gains to the image and return it in RGB
    Call function:
        Parameters:
            item:
                image: image to apply hsv
                labels: labels remain unchanged
        Return:
            image: HSV enhanced image
            labels: original labels
    """
    def __init__(self, h_gain: float, s_gain: float, v_gain: float):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[Image.Image, np.array]:
        image, labels = item

        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        img_hsv = image.convert('HSV')
        hue, sat, val = img_hsv.split()

        x = np.arange(0, 256)
        lut_hue = ((x * r[0]) % 180)
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)

        img_hsv = Image.merge(
            mode='HSV',
            bands=(hue.point(lut_hue), sat.point(lut_sat), val.point(lut_val))
        ).convert('RGB')

        return img_hsv, labels
