# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# The following functions are modified from original AutoML source.
# Changes are identified within the function docstring.

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

import dataloader
import utils
from tf2 import label_util, postprocess
from visualize import vis_utils


def visualize_image(
    image,
    boxes,
    classes,
    scores,
    label_map=None,
    min_score_thresh=0.01,
    max_boxes_to_draw=1000,
    line_thickness=2,
    **kwargs,
):
    """Visualizes a given image.

    Original implementation in tf2/inference.py

    Args:
      image: a image with shape [H, W, C].
      boxes: a box prediction with shape [N, 4] ordered [ymin, xmin, ymax, xmax].
      classes: a class prediction with shape [N].
      scores: A list of float value with shape [N].
      label_map: a dictionary from class id to name.
      min_score_thresh: minimal score for showing. If claass probability is below
        this threshold, then the object will not show up.
      max_boxes_to_draw: maximum bounding box to draw.
      line_thickness: how thick is the bounding box line.
      **kwargs: extra parameters.

    Returns:
      output_image: an output image with annotated boxes and classes.
    """
    label_map = label_util.get_label_map(label_map or "coco")
    category_index = {k: {"id": k, "name": label_map[k]} for k in label_map}
    img = np.array(image)
    vis_utils.visualize_boxes_and_labels_on_image_array(
        img,
        boxes,
        classes,
        scores,
        category_index,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw,
        line_thickness=line_thickness,
        **kwargs,
    )
    return img


def preprocess_resize(imgs: tf.Tensor, image_size: Tuple):
    """Change from original model: Split preprocessing - resize off device, store as uint8,
    normalise on-device, cast to float. This function only handles the resize/crop.
    Modified from the original implementation: keras/efficientdet_keras.py: EfficientDetModel._preprocessing."""
    image_size = utils.parse_image_size(image_size)

    def map_fn(image):
        input_processor = dataloader.DetectionInputProcessor(image, image_size)
        input_processor.set_scale_factors_to_output_size()
        image = input_processor.resize_and_crop_image()
        image_scale = input_processor.image_scale_to_original
        image = tf.cast(image, tf.uint8)
        return image, image_scale

    if imgs.shape.as_list()[0]:  # fixed batch size.
        micro_batch_size = imgs.shape.as_list()[0]
        outputs = [map_fn(imgs[i]) for i in range(micro_batch_size)]
        return [tf.stack(y) for y in zip(*outputs)]

    # otherwise treat it as dynamic batch size.
    return tf.vectorized_map(map_fn, imgs)


def preprocess_normalize_image(img: tf.Tensor, img_dtype: tf.DType):
    """Normalize the image to zero mean and unit variance.
    Original implementation in dataloader.py: InputProcessor"""
    # The image normalization is identical to Cloud TPU ResNet.

    # Equivalent of tf.image.convert_image_dtype from uint8 to float
    img = tf.cast(img, dtype=img_dtype)
    img /= 255.0

    offset = tf.constant([0.485, 0.456, 0.406], dtype=img_dtype)
    offset = tf.broadcast_to(offset, tf.shape(img))
    img -= offset

    scale = tf.constant([0.229, 0.224, 0.225], dtype=img_dtype)
    scale = tf.broadcast_to(scale, tf.shape(img))
    img /= scale
    return img


def postprocess_predictions(config, cls_outputs, box_outputs, scales, mode="global"):
    """Postprocess class and box predictions.
    Modified from the original implementation: keras/efficientdet_keras.py: EfficientDetModel._postprocessing."""
    if not mode:
        return cls_outputs, box_outputs

    # TODO(tanmingxing): remove this cast once FP16 works postprocessing.
    cls_outputs = [tf.cast(i, tf.float32) for i in cls_outputs]
    box_outputs = [tf.cast(i, tf.float32) for i in box_outputs]

    if mode == "global":
        return postprocess.postprocess_global(config.as_dict(), cls_outputs, box_outputs, scales)
    if mode == "per_class":
        return postprocess.postprocess_per_class(config.as_dict(), cls_outputs, box_outputs, scales)
    raise ValueError("Unsupported postprocess mode {}".format(mode))


def visualise_detections(args, config, imgs, inputs, det_outputs, batch_num=0):
    """Modified from the original implementation: inference.py."""
    boxes, scores, classes, valid_len = det_outputs
    os.makedirs(args.output_dir, exist_ok=True)

    start_idx = batch_num * args.micro_batch_size
    end_idx = start_idx + args.micro_batch_size + 1

    for i, img in enumerate(imgs[start_idx:end_idx]):
        length = valid_len[i]
        img = visualize_image(
            img,
            boxes[i].numpy()[:length],
            classes[i].numpy().astype(np.int)[:length],
            scores[i].numpy()[:length],
            label_map=config.label_map,
            min_score_thresh=config.nms_configs.score_thresh,
            max_boxes_to_draw=config.nms_configs.max_output_size,
        )

        def get_filename(x):
            return os.path.join(args.output_dir, f"{args.model_name}_{config.image_size}_{x}_b{batch_num}_img{i}.jpg")

        Image.fromarray(img.astype(np.uint8)).save(get_filename("output"))
        if inputs is not None:
            Image.fromarray(inputs[i].numpy().astype(np.uint8)).save(get_filename("input"))
        print(f"Writing annotated image to {get_filename('output')}")
