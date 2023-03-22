# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from absl import logging
from tensorflow.python import ipu

from hparams_config import Config
from tf2.postprocess import (
    CLASS_OFFSET,
    clip_boxes,
    pre_nms,
    to_list,
)

T = tf.Tensor  # a shortcut for typing check.

BASE_PATH = (Path(__file__).parent / "NMS").absolute()


def nms_op(
    threshold: float,
    score_threshold: float,
    num_detections: int,
    scores: T,
    boxes: T,
    classes: Optional[T] = None,
    multi_nms: bool = False,
) -> Tuple[T, T, T, T, T]:

    attributes = {"threshold": threshold, "scoreThreshold": score_threshold, "numDetections": num_detections}
    attributes_json = json.dumps(attributes)
    logging.debug(attributes_json)

    output_shape = [scores.shape[0], num_detections]

    outputs = {
        "output_types": [tf.int32, tf.float16, tf.float16, tf.int32, tf.int32],
        "output_shapes": [
            tf.TensorShape(output_shape),
            tf.TensorShape(output_shape),
            tf.TensorShape([scores.shape[0], num_detections, 4]),
            tf.TensorShape(output_shape),
            tf.TensorShape([scores.shape[0]]),
        ],
    }

    if multi_nms:
        inputs = [scores, boxes]
    elif classes is None:
        raise RuntimeError("NMS requires the classes unless running in Multiclass mode.")
    else:
        scores = tf.reduce_max(scores, -1)
        inputs = [scores, boxes, tf.cast(classes, tf.int32)]

    nms_type_str = "tf_multi" if multi_nms else "tf"
    lib_path = BASE_PATH / nms_type_str / "build" / "nms_custom_op.so"
    gp_path = BASE_PATH / "codelet.cpp"

    return ipu.custom_ops.precompiled_user_op(
        inputs, str(lib_path), str(gp_path), attributes=attributes_json, outs=outputs
    )


def ipu_nms(params, scores: T, boxes: T, classes: T, multi_nms: bool = False) -> Tuple[T, T, T, T]:
    """Non-maximum suppression.

    Args:
      params: a dict of parameters.
      boxes: a tensor with shape [N, 4], where N is the number of boxes. Box
        format is [y_min, x_min, y_max, x_max].
      scores: a tensor with shape [N].
      classes: a tensor with shape [N].

    Returns:
      A tuple (boxes, scores, classes, valid_lens), where valid_lens is a scalar
      denoting the valid length of boxes/scores/classes outputs.
    """
    nms_configs = params["nms_configs"]
    max_output_size = nms_configs["max_output_size"]
    iou_thresh = nms_configs["iou_thresh"]
    score_thresh = nms_configs["score_thresh"]

    nms_idx, nms_scores, nms_boxes, nms_classes, nms_lengths = nms_op(
        threshold=iou_thresh,
        score_threshold=score_thresh,
        num_detections=max_output_size,
        scores=scores,
        boxes=boxes,
        classes=classes,
        multi_nms=multi_nms,
    )

    nms_classes = nms_classes + CLASS_OFFSET

    return nms_boxes, nms_scores, nms_classes, nms_idx


def ipu_postprocessing(
    config: Config, step_outputs: List[T], image_scales: T = None, multi_nms: bool = False
) -> Tuple[T, T, T, T]:
    cls_outputs, box_outputs = step_outputs

    cls_outputs = to_list(cls_outputs)
    box_outputs = to_list(box_outputs)

    boxes, scores, classes = pre_nms(config.as_dict(), cls_outputs, box_outputs, topk=False)

    outputs = ipu_nms(config.as_dict(), scores, boxes, classes, multi_nms)
    nms_boxes, nms_scores, nms_classes, nms_idx = outputs

    nms_boxes = clip_boxes(nms_boxes, config.image_size)
    if image_scales is not None:
        scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
        nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
    return nms_boxes, nms_scores, nms_classes, nms_idx


def postprocess_onchip_nms_outputs(config: Config, det_outputs: List[np.array]) -> List[np.array]:
    nms_boxes, nms_scores, nms_classes, nms_idx = det_outputs
    valid_len = np.argmax(nms_idx == -1, axis=1)
    all_valid = np.logical_not(np.any(nms_idx == -1, axis=1))
    valid_len[all_valid] = config.nms_configs.max_output_size
    return [nms_boxes, nms_scores, nms_classes, valid_len]
