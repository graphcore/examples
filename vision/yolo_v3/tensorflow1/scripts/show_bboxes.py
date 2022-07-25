#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

import argparse

import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--line-id", default=0)
parser.add_argument("--annotation-path", default="./data/dataset/voc_train.txt")
arguments = parser.parse_args()
with open(arguments.annotation_path) as f:
    image_info = f.readlines()[arguments.line_id].split()

image_path = image_info[0]
image = cv2.imread(image_path)
for bbox in image_info[1:]:
    bbox = bbox.split(",")
    image = cv2.rectangle(image, (int(float(bbox[0])),
                                  int(float(bbox[1]))),
                          (int(float(bbox[2])),
                           int(float(bbox[3]))), (255, 0, 0), 2)

cv2.imshow('image', np.uint8(image))
cv2.waitKey(0)
