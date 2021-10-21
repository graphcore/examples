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

from typing import List


def plotting_tool(cfg: CfgNode, pruned_preds_batch: List[np.array], orig_img: List[Image.Image]):
    """
    Plots the predicted bounding boxes and class labels on top of the original image
    Parameters:
        cfg: yacs object containing the config
        pruned_preds_batch (List[np.array]): bounding box predictions
        orig_img (List[Image.Image]): original image
    """
    class_names = yaml.safe_load(open(cfg.model.class_name_path))['class_names']

    for i, pruned_preds in enumerate(pruned_preds_batch):
        bboxes = pruned_preds[:, :4]
        class_pred = pruned_preds[:, 5].astype(int)
        plot_img_bboxes(
            bboxes, class_pred, orig_img[i], class_names, cfg.inference.plot_dir
        )


def scale_boxes_to_orig(boxes: np.array, orig_h: int, orig_w: int, img_size: int) -> np.array:
    """
    Scale the bounding boxes returned from the model to the image original size and remove padding
    Parameters:
        boxes (np.array): an array containing bboxes in (centerx, centery, width, height)
        orig_h (int): original image height
        orig_w (int): orignal image width
    Return:
        (np.array): An array conains bboxes in (x, y, w, h) in the scale of the original image
    """
    ratio_image_size = img_size / max(orig_h, orig_w)
    pad_width = int(np.round(abs(img_size-(ratio_image_size * orig_w))/2))
    pad_height = int(np.round(abs(img_size-(ratio_image_size * orig_h))/2))

    boxes[:, 0] -= pad_width
    boxes[:, 1] -= pad_height

    boxes[:, [0, 2]] /= ratio_image_size
    boxes[:, [1, 3]] /= ratio_image_size

    return boxes


def plot_img_bboxes(
    bboxes: np.array, class_pred: np.array, orig_img: Image.Image, class_names: List[str], image_dir: str = ""
) -> str:
    """
    Plot the image with bouding boxes around detections.
    Parameters:
        bboxes (np.array): an array of bounding boxes in the original image size
        class_pred (np.array): label of the object associated with each bounding box
        orig_img (Image.Image): the original input image to plot
        class_names (List[str]): string label
        image_dir (str): optional, if set, the image will be saved to the directory
    Return:
        File path of the saved image, empty if not saved
    """
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names) * 50)]
    random.shuffle(colors)

    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(orig_img)

    for i, (xcenter, ycenter, box_w, box_h) in enumerate(bboxes):
            cls = class_pred[i]
            xmin = xcenter - box_w / 2
            ymin = ycenter - box_h / 2
            bbox = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=2, edgecolor=colors[cls], facecolor="none")
            ax.add_patch(bbox)
            plt.text(
                xmin,
                ymin,
                s=class_names[cls],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[cls], "pad": 0}
            )
    plt.show()

    if image_dir:
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)
        file_name = "detection_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-4])
        image_dir = "{}/{}".format(image_dir, file_name)
        fig.savefig(image_dir)

    plt.close()
    return image_dir
