# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from datetime import datetime
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import random
from ruamel import yaml
from PIL import Image
from yacs.config import CfgNode
from typing import List, Union
import torch


def plotting_tool(cfg: CfgNode, pruned_preds_batch: List[np.array], orig_img: List[Image.Image]):
    """
    Plots the predicted bounding boxes and class labels on top of the original image
    Parameters:
        cfg: yacs object containing the config
        pruned_preds_batch (List[np.array]): bounding box predictions
        orig_img (List[Image.Image]): original image
    """
    class_names = yaml.safe_load(open(cfg.model.class_name_path))["class_names"]

    img_paths = []
    for i, pruned_preds in enumerate(pruned_preds_batch):
        idx = pruned_preds[:, 4] > cfg.inference.plot_threshold
        pruned_preds = pruned_preds[idx]
        bboxes = pruned_preds[:, :4]
        class_pred = pruned_preds[:, 5].int()
        path = plot_img_bboxes(bboxes, class_pred, orig_img[i], class_names, cfg.inference.plot_dir)
        img_paths.append(path)
    return img_paths


def scale_boxes_to_orig(
    boxes: Union[np.array, torch.Tensor], original_sizes: torch.Tensor, img_size: torch.Tensor
) -> Union[np.array, torch.Tensor]:
    """
    Scale the bounding boxes returned from the model to the image original size and remove padding
    Parameters:
        boxes (np.array or torch.Tensor): an array containing bboxes in (centerx, centery, width, height)
        original_sizes (torch.Tensor): original image height and width
        img_size (torch.Tensor): original image size
    Return:
        (np.array or torch.Tensor): An array contains bboxes in (x, y, w, h) in the scale of the original image
    """
    xy, wh = boxes[..., 0:2], boxes[..., 2:4]

    ratio_image_size = img_size / (original_sizes).max().unsqueeze(axis=-1).float()
    padding = ((img_size - (ratio_image_size * original_sizes).float()).abs() / 2).int()

    if type(boxes).__module__ == np.__name__:
        xy = xy - padding.numpy()
        new_boxes = np.concatenate((xy, wh), axis=-1) / ratio_image_size

        if boxes.shape[-1] != 4:
            new_boxes = np.concatenate((new_boxes, boxes[..., 4:]), axis=-1)
    else:
        xy = xy - padding
        new_boxes = torch.cat((xy, wh), axis=-1) / ratio_image_size

        if boxes.shape[-1] != 4:
            new_boxes = torch.cat((new_boxes, boxes[..., 4:]), axis=-1)

    return new_boxes


def plot_img_bboxes(
    bboxes: np.array, class_pred: np.array, orig_img: Image.Image, class_names: List[str], image_dir: str = ""
) -> str:
    """
    Plot the image with bounding boxes around detections.
    Parameters:
        bboxes (np.array): an array of bounding boxes in the original image size
        class_pred (np.array): label of the object associated with each bounding box
        orig_img (Image.Image): the original input image to plot
        class_names (List[str]): string label
        image_dir (str): optional, if set, the image will be saved to the directory
    Return:
        File path of the saved image, empty if not saved
    """
    n_classes = len(class_names)

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, n_classes * 50)]

    random.shuffle(colors)

    plt.figure(dpi=400)

    fig, ax = plt.subplots()
    ax.imshow(orig_img)
    # remove axes from figure
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.rc("font", size=8)

    for i, (xcenter, ycenter, box_w, box_h) in enumerate(bboxes):
        class_number = class_pred[i]
        empty_detection = class_number > n_classes or class_number < 0
        if not empty_detection:
            xmin = xcenter - box_w / 2
            ymin = ycenter - box_h / 2
            bbox = patches.Rectangle(
                (xmin, ymin), box_w, box_h, linewidth=1, edgecolor=colors[class_number], facecolor="none"
            )
            ax.add_patch(bbox)
            plt.text(
                xmin,
                ymin,
                s=class_names[class_number],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[class_number], "pad": 0},
            )
    plt.show()

    if image_dir:
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)
        file_name = "detection_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-4])
        image_dir = "{}/{}".format(image_dir, file_name)
        fig.savefig(image_dir, dpi=400, bbox_inches="tight", pad_inches=0)

    plt.close()
    return image_dir
