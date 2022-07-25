#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

import argparse
import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm


def main(json_path, image_path, coco_categories, coco_names, output_annotation):

    dataset = defaultdict(list)

    # coco has 80 categories but it's id is not continuous
    # it's category ids go up to 90
    # we want to map these ids to 0-79
    # so that we won't need a model head that has 90 classes
    coco_90_to_80 = {}
    with open(coco_categories, "r") as f:
        coco_original_dict = {}
        for line in f:
            category_id, name = line.strip().split(",")
            coco_original_dict[name] = int(category_id)
    with open(coco_names, "r") as f:
        for index, line in enumerate(f):
            name = line.strip()
            coco_90_to_80[coco_original_dict[name]] = index

    with open(os.path.realpath(output_annotation), "w") as f:
        labels = json.load(open(json_path, encoding="utf-8"))
        annotations = labels["annotations"]

        for annotation in tqdm(annotations):
            image_id = annotation["image_id"]
            image_folder_path = os.path.realpath(image_path)
            single_image_path = os.path.join(image_folder_path, "%012d.jpg" % image_id)
            category_id_coco = annotation["category_id"]
            category_id = coco_90_to_80[category_id_coco]

            x_min, y_min, width, height = annotation["bbox"]
            x_max = x_min+width
            y_max = y_min+height
            box = [x_min, y_min, x_max, y_max]
            dataset[single_image_path].append([category_id, box])

        for single_image_path in dataset.keys():
            write_content = [single_image_path]

            for category_id, box in dataset[single_image_path]:
                x_min, y_min, x_max, y_max = box
                write_content.append(",".join([str(round(x_min)), str(round(y_min)),
                                               str(round(x_max)), str(round(y_max)), str(category_id)]))

            write_content = " ".join(write_content)
            f.write(write_content+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert coco data to the format of "image_path [xmin,ymin,xmax,ymax,class_id]..."')
    parser.add_argument("--train-json-path", default="./coco/annotations/instances_train2017.json")
    parser.add_argument("--val-json-path", default="./coco/annotations/instances_val2017.json")
    parser.add_argument("--train-image-path", default="./coco/train2017")
    parser.add_argument("--val-image-path", default="./coco/val2017")
    parser.add_argument("--coco-categories", default="./data/classes/coco_categories.txt")
    parser.add_argument("--coco-names", default="./data/classes/coco.names")
    parser.add_argument("--output-train-annotation", default="./data/dataset/coco_train2017.txt")
    parser.add_argument("--output-val-annotation", default="./data/dataset/coco_val2017.txt")
    arguments = parser.parse_args()
    print("generating training annotation")
    main(arguments.train_json_path,
         arguments.train_image_path,
         arguments.coco_categories,
         arguments.coco_names,
         arguments.output_train_annotation)

    print("generating validation annotation")
    main(arguments.val_json_path,
         arguments.val_image_path,
         arguments.coco_categories,
         arguments.coco_names,
         arguments.output_val_annotation)
