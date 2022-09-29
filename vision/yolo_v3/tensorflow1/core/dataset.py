#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

import colorsys
import gc
import os
import pickle
import random
import sys
import threading
import time
from collections import deque

import numpy as np

import core.utils as utils
import cv2
from log import logger

lock = threading.Lock()
lock_load = threading.Lock()


class Dataset(object):
    """Dataset class for loading and preprocessing data"""

    def __init__(self, dataset_type, opts):
        """Prepare info for generating data
        Args:
            dataset_type: one of {"train", "test"}
            opts: configuration of type dict
        """
        self.annot_path = opts[dataset_type]["annot_path"]
        self.batch_size = opts[dataset_type]["batch_size"]
        self.data_aug = opts[dataset_type]["data_aug"]
        self.use_mosaic_input = opts["train"]["use_mosaic_input"]
        self.use_color_augment = opts["train"]["use_color_augment"]

        self.train_input_sizes = opts["train"]["input_size"]
        self.strides = np.array(opts["yolo"]["strides"])
        classes = utils.read_class_names(opts["yolo"]["classes"])
        self.num_classes = len(classes)
        self.anchors = np.array(utils.get_anchors(opts["yolo"]["anchors"]))
        self.anchor_per_scale = opts["yolo"]["anchor_per_scale"]
        self.data_name = opts["yolo"]["data_name"]
        self.max_bbox_per_scale = 150

        self.dataset_type = dataset_type
        self.annotations = self.load_annotations(self.annot_path)
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.dump_files = None
        self.read_index = 0
        self.images = None
        self.cache_images = None
        self.use_pre_load = opts["train"]["use_pre_load"]

    def load_annotations(self, annot_path):
        """Load annotations file
        Annotation file should follow format of "image_path [xmin,ymin,xmax,ymax,class_id]..."
        """

        with open(annot_path, "r") as f:
            txt = f.readlines()
            annotations = [line.strip()
                           for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        """Generate one batch of input data
        Return:
            batch_image: a batch of preprocessed images.
            batch_label_sbbox: Label for every small scale anchor with the shape of
                (batch_size, grid_size, grid_size, anchors_per_point, classes+objectness+bbox_coordinates).
            batch_label_mbbox: Label for every middle scale anchor with the shape of.
            batch_label_lbbox: Label for every large scale anchor with the shape of.
            batch_sbboxes: Small boxes in shape (batch_size, max_box_per_scale, coordinates).
            batch_mbboxes: Middle boxes in shape (batch_size, max_box_per_scale, coordinates).
            batch_lbboxes: Large boxes in shape (batch_size, max_box_per_scale, coordinates).
        """
        self.train_input_size = random.choice(self.train_input_sizes)
        self.train_output_sizes = self.train_input_size // self.strides

        # prepare output numpy arrays
        batch_image = np.zeros(
            (self.batch_size, self.train_input_size, self.train_input_size, 3))

        batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                      self.anchor_per_scale, 5 + self.num_classes))

        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

        # image index in a batch
        index_in_batch = 0
        if self.batch_count < self.num_batches:
            while index_in_batch < self.batch_size:
                if self.use_mosaic_input and random.random() > 0.8:
                    # if use mosaic data
                    # we stitch four images into one
                    # so every image will be 1/4 of the size of network input
                    final_image = np.zeros(
                        (self.train_input_size, self.train_input_size, 3))
                    all_bboxes = []
                    half_image_size = self.train_input_size//2

                    image, bboxes = self.parse_annotation(
                        half_image_size)
                    final_image[:half_image_size, :half_image_size] = image
                    all_bboxes.extend(bboxes)

                    image, bboxes = self.parse_annotation(
                        half_image_size)
                    final_image[:half_image_size, half_image_size:] = image
                    all_bboxes.extend(
                        [[x_1+half_image_size, y_1, x_2+half_image_size, y_2, class_index]
                            for x_1, y_1, x_2, y_2, class_index in bboxes])

                    image, bboxes = self.parse_annotation(
                        half_image_size)
                    final_image[half_image_size:, :half_image_size] = image
                    all_bboxes.extend(
                        [[x_1, y_1+half_image_size, x_2, y_2+half_image_size, class_index]
                            for x_1, y_1, x_2, y_2, class_index in bboxes])

                    image, bboxes = self.parse_annotation(
                        half_image_size)
                    final_image[half_image_size:, half_image_size:] = image
                    all_bboxes.extend(
                        [[x_1+half_image_size, y_1+half_image_size, x_2+half_image_size, y_2+half_image_size, class_index]
                            for x_1, y_1, x_2, y_2, class_index in bboxes])

                    image = final_image
                    bboxes = np.array(all_bboxes)
                else:
                    # if use only one image
                    image, bboxes = self.parse_annotation(
                        self.train_input_size)
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                    bboxes)

                batch_image[index_in_batch, :, :, :] = image
                batch_label_sbbox[index_in_batch, :, :, :, :] = label_sbbox
                batch_label_mbbox[index_in_batch, :, :, :, :] = label_mbbox
                batch_label_lbbox[index_in_batch, :, :, :, :] = label_lbbox
                batch_sbboxes[index_in_batch, :, :] = sbboxes
                batch_mbboxes[index_in_batch, :, :] = mbboxes
                batch_lbboxes[index_in_batch, :, :] = lbboxes
                index_in_batch += 1
            self.batch_count += 1

            return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                batch_sbboxes, batch_mbboxes, batch_lbboxes
        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        """randomly crop image
        This won't remove or clip part of bboxes, only remove parts without box
        """

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def adjust_color(self, image):
        """adjust image hue, saturation, lightness randomly
        """
        # convert color space to hls
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l = np.random.randint(0, 2)+0.5
        s = np.random.randint(0, 2)+0.5
        h = np.random.randint(0, 2)*0.2+0.9
        if np.random.random() > 0.5:
            image[:, :, 0] = h * image[:, :, 0]

        # adjust saturation
        if np.random.random() > 0.5:
            image[:, :, 1] = l * image[:, :, 1]
        # adjust saturation
        if np.random.random() > 0.5:
            image[:, :, 2] = s * image[:, :, 2]
        # HLS2BGR
        image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
        return image

    def get_next_data(self):
        """take an image form self.images
        This function will also load or preload image chunks according to configuration.
        """
        def get_next_pair():
            key = self.keys[self.read_index]
            image = self.images[key]
            self.read_index += 1
            return image, key

        def try_load_cache():
            if self.cache_images is None and lock_load.acquire(blocking=False):
                self.cache_images = self.load_images()
                lock_load.release()
                return True
            return False

        if self.use_pre_load:
            # if pre_load use try this
            # other way, we load data when there's on more images to process
            try_load_cache()

        lock.acquire()
        while True:
            data_remains = self.images is not None and self.read_index < len(self.images)
            if data_remains:
                # if there is data loaded
                image, key = get_next_pair()
                break
            else:
                # if it reach this line
                # that means self.images is useless anyway
                if self.images is not None:
                    self.images = None
                    gc.collect()
                if not try_load_cache():
                    time.sleep(0.01)
                if self.cache_images is not None:
                    # put cache into current using images
                    self.images = self.cache_images
                    self.cache_images = None
                    self.read_index = 0
                    self.keys = list(self.images.keys())
                    np.random.shuffle(self.keys)

        lock.release()

        return image, key

    def parse_annotation(self, target_image_size):
        """Read and preprocess Image.
        Take an image and randomly augment image and resize image to target_image_size in height and width.
        """
        image, annotation = self.get_next_data()
        line = annotation.split()

        bboxes = np.array(
            [list(map(lambda x: int(float(x)), box.split(","))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes))
            if self.use_color_augment:
                image = self.adjust_color(image)

        image, bboxes = utils.resize_image(
            np.copy(image), [target_image_size, target_image_size], np.copy(bboxes))

        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        """Preprocess bboxes.
        Classify boxes into small, middle, large boxes and generate per anchor label.
        Args:
            bboxes: groundtruth boxes, shape: (number_boxes_of_image, 4_coordinates+class_id)
        Returns:
            label_sbbox: Small boxes label for every anchor.
                shape: (grid_size, grid_size, anchors_per_point, classes+objectness+bbox_coordinates)
            label_mbbox: Middle boxes label for every anchor.
                shape: (grid_size, grid_size, anchors_per_point, classes+objectness+bbox_coordinates)
            label_lbbox: Large boxes label for every anchor.
                shape: (grid_size, grid_size, anchors_per_point, classes+objectness+bbox_coordinates)
            sbboxes: Small boxes in shape (max_box_per_scale, coordinates).
            mbboxes: Middle boxes in shape (max_box_per_scale, coordinates).
            lbboxes: Large boxes in shape (max_box_per_scale, coordinates).
        """

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4))
                       for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * \
                bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(
                    bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(
                        bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] %
                               self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batches

    def load_images(self, dump=True):
        """Generating images thunks and Load chunks.
        Args:
            dump: if dump chunks for caching.
        """
        dump_dir = "./data/"+self.data_name + "_" + self.dataset_type
        dump_pattern = dump_dir+"/"+"pickle_{:03d}"

        def load_data():
            if self.dump_files is None:
                self.dump_files = os.listdir(dump_dir)
                self.file_index = 0
                # shuffle dump_files to make the sequence different every time
                # or different for every instance when using popdist or using multiprocess
                np.random.shuffle(self.dump_files)
            file_path = dump_dir+"/"+self.dump_files[self.file_index]
            logger.info("loading file: {}".format(file_path))
            with open(file_path, "rb") as f:
                self.file_index = (self.file_index+1) % len(self.dump_files)
                if self.file_index == 0:
                    # file_index == 0 means that all files have been read for once
                    # we reshuffle it
                    np.random.shuffle(self.dump_files)
                # use pickle.load will cause memory leak
                # so we read it in our code and the use pickle.loads
                content = f.read()
                images = pickle.loads(content)
                del content
            return images

        def dump_file(dump_path, images):
            with open(dump_path, "wb") as f:
                if dump:
                    pickle.dump(images, f)

        def generate_data():
            logger.info("generating data...")
            images = {}
            dump_count = 0
            if dump and not os.path.exists(dump_dir):
                os.mkdir(dump_dir)
            for annotation in self.annotations:
                line = annotation.split()
                image_path = line[0]
                if not os.path.exists(image_path):
                    raise KeyError("%s does not exist ... " % image_path)
                image = np.array(cv2.imread(image_path))
                images[annotation] = image
                if len(images) >= 2000 and dump:
                    dump_file(dump_pattern.format(dump_count), images)
                    dump_count += 1
                    images = {}
            if len(images) > 0 and dump:
                dump_file(dump_pattern.format(dump_count), images)

            return images

        logger.info("loading images done")

        if not dump or not os.path.exists(dump_dir) or len(os.listdir(dump_dir)) == 0:
            images = generate_data()
        if not dump:
            return images
        else:
            return load_data()
