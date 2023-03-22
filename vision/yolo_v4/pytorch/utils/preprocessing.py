# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import List, Tuple

import math
import numpy as np
import torch
from PIL import Image

from utils.tools import ioa, normalize_labels, standardize_labels, xywh_to_xyxy, xyxy_to_xywh

"""
Torch transformations for the YOLOv4 dataset
"""


class PadNumpy(object):
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

        pad_width = int(np.round(abs(self.image_size - input_image.shape[1]) / 2))
        pad_height = int(np.round(abs(self.image_size - input_image.shape[0]) / 2))

        output_image[
            pad_height : (input_image.shape[0] + pad_height), pad_width : (input_image.shape[1] + pad_width), :
        ] = input_image

        labels[:, [1, 3]] *= input_image.shape[1] / self.image_size
        labels[:, [2, 4]] *= input_image.shape[0] / self.image_size
        labels[:, 1] += (pad_width) / self.image_size
        labels[:, 2] += (pad_height) / self.image_size

        return output_image, labels


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

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[Image.Image, np.array]:
        input_image, labels = item

        pad_width = int(np.round(abs(self.image_size - input_image.width) / 2))
        pad_height = int(np.round(abs(self.image_size - input_image.height) / 2))

        output_image = Image.new(input_image.mode, (self.image_size, self.image_size), (114, 114, 114))
        output_image.paste(input_image, (pad_width, pad_height))

        labels[:, [1, 3]] *= input_image.width / self.image_size
        labels[:, [2, 4]] *= input_image.height / self.image_size
        labels[:, 1] += (pad_width) / self.image_size
        labels[:, 2] += (pad_height) / self.image_size

        return output_image, labels


class ResizeImage(object):
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

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[Image.Image, np.array]:
        image, labels = item

        height, width = image.size

        ratio_image_size = self.image_size / max(height, width)
        if ratio_image_size != 1:
            interp = Image.BILINEAR if ratio_image_size > 1 else Image.NEAREST
            return image.resize((int(height * ratio_image_size), int(width * ratio_image_size)), interp), labels
        else:
            return image, labels


class ToNumpy(object):
    """
    Convert PIL.Image.Image to numpy object
    Call function:
        Parameters:
            item:
                image: PIL Image to convert to
                labels: labels doesn't perform any transformation
        Return:
            image: numpy image
            labels: same labels
    """

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[np.array, np.array]:
        image, labels = item
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
        output_labels[: input_labels.shape[0]] = torch.from_numpy(input_labels)

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.image_type == "half":
            image = image.half() / 255.0
        elif self.image_type == "float":
            image = image.float() / 255.0
        return image, output_labels


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
        self.scaled_threshold = scaled_threshold

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
            if len(labels) and scale > self.scaled_threshold:
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
        img_hsv = image.convert("HSV")
        hue, sat, val = img_hsv.split()

        x = np.arange(0, 256)
        lut_hue = (x * r[0]) % 180
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)

        img_hsv = Image.merge(mode="HSV", bands=(hue.point(lut_hue), sat.point(lut_sat), val.point(lut_val))).convert(
            "RGB"
        )

        return img_hsv, labels


class Mosaic(object):
    """
    Apply Mosaic by combining the current image with 3 other random images
    Call function:
        Parameters:
            item:
                4 pairs of images and labels that will be stitched togeter. The first image
                is the current image while the other 3 are randomly selected
        Return:
            image: combined target image with other 3 images
            labels: combined labels from the target images with the other 3
    """

    def __init__(self, input_image_size: int, input_channels: int):
        self.image_size = input_image_size
        self.input_channels = input_channels

    def __call__(self, item: Tuple[Tuple[Image.Image, np.array]]) -> Tuple[Image.Image, np.array]:

        # base image of size self.image_size * 2 x self.image_size * 2
        base_img = np.full((self.image_size * 2, self.image_size * 2, self.input_channels), 114)
        base_label = []
        # each grid in the base_img will be of size self.image_size x self.image_size
        center_x, center_y = self.image_size, self.image_size

        for i, (image, label) in enumerate(item):
            original_image = np.ascontiguousarray(image)
            height, width = original_image.shape[0], original_image.shape[1]

            if i == 0:  # top left
                # the top-left location on the base_img
                xmin_b, ymin_b, xmax_b, ymax_b = max(center_x - width, 0), max(center_y - height, 0), center_x, center_y
                # location on the original image to copy to the base_img
                xmin_o, ymin_o, xmax_o, ymax_o = width - (xmax_b - xmin_b), height - (ymax_b - ymin_b), width, height
            elif i == 1:  # top right
                xmin_b, ymin_b, xmax_b, ymax_b = (
                    center_x,
                    max(center_y - height, 0),
                    min(center_x + width, self.image_size * 2),
                    center_y,
                )
                xmin_o, ymin_o, xmax_o, ymax_o = 0, height - (ymax_b - ymin_b), min(width, xmax_b - xmin_b), height
            elif i == 2:  # bottom left
                xmin_b, ymin_b, xmax_b, ymax_b = (
                    max(center_x - width, 0),
                    center_y,
                    center_x,
                    min(self.image_size * 2, center_y + height),
                )
                xmin_o, ymin_o, xmax_o, ymax_o = (
                    width - (xmax_b - xmin_b),
                    0,
                    max(center_x, width),
                    min(ymax_b - ymin_b, height),
                )
            elif i == 3:  # bottom right
                xmin_b, ymin_b, xmax_b, ymax_b = (
                    center_x,
                    center_y,
                    min(center_x + width, self.image_size * 2),
                    min(self.image_size * 2, center_y + height),
                )
                xmin_o, ymin_o, xmax_o, ymax_o = 0, 0, min(width, xmax_b - xmin_b), min(ymax_b - ymin_b, height)

            base_img[ymin_b:ymax_b, xmin_b:xmax_b] = original_image[ymin_o:ymax_o, xmin_o:xmax_o]
            padw = xmin_b - xmin_o
            padh = ymin_b - ymin_o

            # Labels
            if label.size > 0:
                # Normalized xywh to pixel xywh format in order to shift xy by the padding size
                label = standardize_labels(label, width, height)
                label[:, 1] = label[:, 1] + padw
                label[:, 2] = label[:, 2] + padh
                # Divide by the new image size to the new normalize xywh
                label[:, 1:] = label[:, 1:] / (self.image_size * 2)
            base_label.append(label)

        if len(base_label):
            base_label = np.concatenate(base_label, 0)
        np.clip(base_label[:, 1:], 0, self.image_size, out=base_label[:, 1:])

        return Image.fromarray(np.uint8(base_img)), base_label


class RandomPerspective(object):
    """
    Apply transformations, including perspective, rotation, translation, shear on the target image
    Call function:
        Parameters:
            item:
                image: image to apply the transformation
                labels: labels that will be transformed with the image
        Return:
            image: image after affine transform has been applied
            labels: labels of remaining objects after the transformation
    """

    def __init__(
        self,
        degrees: int = 10,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: int = 10,
        perspective: float = 0.0,
        border_to_remove: Tuple[int, int] = (0, 0),
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border_to_remove = border_to_remove

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[Image.Image, np.array]:
        image, labels = item

        orig_height, orig_width = image.size[0], image.size[1]
        target_height = orig_height - self.border_to_remove[0]
        target_width = orig_width - self.border_to_remove[1]

        # Center
        center = np.eye(3)
        center[0, 2] = -image.size[1] / 2  # x translation (pixels)
        center[1, 2] = -image.size[0] / 2  # y translation (pixels)

        # Perspective
        perspective = np.eye(3)
        perspective[2, 0] = np.random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        perspective[2, 1] = np.random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        rotation = np.eye(3)
        angle = np.random.uniform(-self.degrees, self.degrees)
        scale = np.random.uniform(1 - self.scale, 1 + self.scale)
        # implementation https://docs.opencv.org/4.5.3/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
        alpha = scale * math.cos(angle)
        beta = scale * math.sin(angle)
        # set center = (0, 0)
        rotation[:2] = np.array([[alpha, beta, 0], [-beta, alpha, 0]])

        # Shear
        shear = np.eye(3)
        shear[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        shear[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        translation = np.eye(3)
        translation[0, 2] = (
            np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * target_width
        )  # x translation (pixels)
        translation[1, 2] = (
            np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * target_height
        )  # y translation (pixels)

        # Combined rotation matrix
        combined_matrix = translation @ shear @ rotation @ perspective @ center
        if (
            (self.border_to_remove[0] != 0) or (self.border_to_remove[1] != 0) or (combined_matrix != np.eye(3)).any()
        ):  # check if image changed
            combined_matrix_inv = np.linalg.inv(combined_matrix)
            image = image.transform(
                (target_width, target_height),
                Image.AFFINE,
                combined_matrix_inv.flatten()[:6],
                fillcolor=(114, 114, 114),
            )

        if len(labels):
            n = len(labels)

            # Normalized xywh to pixel xywh format in order to manipulate the label
            labels = standardize_labels(labels, orig_width, orig_height)
            # Convert to xyxy in order to perform transformation
            labels[:, 1:] = xywh_to_xyxy(labels[:, 1:])

            xy = np.ones((n * 4, 3))
            xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ combined_matrix.T
            xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, target_width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, target_height)

            # filter candidates
            filter_indices = self.box_candidates(labels[:, 1:].T * scale, xy.T)
            labels = labels[filter_indices]
            labels[:, 1:] = xy[filter_indices]

            # convert labels back to normalized label
            labels[:, 1:] = xyxy_to_xywh(labels[:, 1:])
            labels = normalize_labels(labels, target_width, target_height)

        return image, labels

    def box_candidates(
        self,
        boxes1: np.array,
        boxes2: np.array,
        wh_threshold: float = 2.0,
        aspect_ratio_threshold: float = 20.0,
        area_threshold: float = 0.2,
    ) -> np.array:
        """
        Remove small boxes by compare the candidate boxes1 (before augment) and boxes2 (after augment)
        if they have width/height threshold (pixels), aspect_ratio_threshold and area_threshold more than specified
        # Parameters:
            boxes1 (np.array): 4xN labels before augmentation
            boxes2 (np.array): 4xN labels after augmentation
            wh_threshold (float): threshold to remove any boxes that are smaller (in absolute pixels) than the threshold
            aspect_ratio_threshold (float): aspect ratio of the new boxes, in order to get rid of edges or labels that are too narrow
            area_threshold (float): area threshold between the new boxes and the old boxes
        # Returns:
            np.array: 4xM incides of the boxes that pass the threshold
        """
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
        aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (
            (w2 > wh_threshold)
            & (h2 > wh_threshold)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_threshold)
            & (aspect_ratio < aspect_ratio_threshold)
        )


class HorizontalFlip(object):
    """
    Flip the image and labels horizontally
    Call Function:
        Parameters:
            item:
                image: image to flip
                labels: labels for the image
        Return:
            image: flipped image
            labels: flipped labels
    """

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[Image.Image, np.array]:
        image, labels = item
        output_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        labels[:, 1] = 1 - labels[:, 1]
        return output_image, labels


class VerticalFlip(object):
    """
    Flip the image and labels vertically
    Call Function:
        Parameters:
            item:
                image: image to flip
                labels: labels for the image
        Return:
            image: flipped image
            labels: flipped labels
    """

    def __call__(self, item: Tuple[Image.Image, np.array]) -> Tuple[Image.Image, np.array]:
        image, labels = item
        output_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        labels[:, 2] = 1 - labels[:, 2]
        return output_image, labels
