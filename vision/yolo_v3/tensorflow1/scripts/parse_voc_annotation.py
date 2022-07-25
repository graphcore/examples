#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

import argparse
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = []
    with open("./data/classes/voc.names") as f:
        for line in f:
            classes.append(line.strip())

    img_inds_file = os.path.join(data_path, "ImageSets", "Main", data_type + ".txt")
    with open(img_inds_file, "r") as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, "a") as f:
        for image_ind in tqdm(image_inds):
            image_path = os.path.join(data_path, "JPEGImages", image_ind + ".jpg")
            annotation = image_path
            label_path = os.path.join(data_path, "Annotations", image_ind + ".xml")
            root = ET.parse(label_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                bbox = obj.find("bndbox")
                class_ind = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find("xmin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymin = bbox.find("ymin").text.strip()
                ymax = bbox.find("ymax").text.strip()
                annotation += " " + ",".join([xmin, ymin, xmax, ymax, str(class_ind)])
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert voc data to the format of "image_path [xmin,ymin,xmax,ymax,class_id]..."')
    parser.add_argument("--data_path", default="./VOC/")
    parser.add_argument("--train_annotation", default="./data/dataset/voc_train.txt")
    parser.add_argument("--test_annotation",  default="./data/dataset/voc_test.txt")
    arguments = parser.parse_args()

    if os.path.exists(arguments.train_annotation):
        os.remove(arguments.train_annotation)
    if os.path.exists(arguments.test_annotation):
        os.remove(arguments.test_annotation)

    print("generating 2007 training data")
    num1 = convert_voc_annotation(os.path.join(arguments.data_path, "train/VOCdevkit/VOC2007"),
                                  "trainval", arguments.train_annotation, False)
    print("generating 2012 training data")
    num2 = convert_voc_annotation(os.path.join(arguments.data_path, "train/VOCdevkit/VOC2012"),
                                  "trainval", arguments.train_annotation, False)
    print("generating 2007 testing data")
    num3 = convert_voc_annotation(os.path.join(arguments.data_path, "test/VOCdevkit/VOC2007"),
                                  "test", arguments.test_annotation, False)
    print("=> The number of image for train is: %d\tThe number of image for test is:%d" % (num1 + num2, num3))
