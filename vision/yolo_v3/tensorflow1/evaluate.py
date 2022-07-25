#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
# Copyright (c) 2019 YunYang1994 <dreameryangyun@sjtu.edu.cn>
# License: MIT (https://opensource.org/licenses/MIT)
# This file has been modified by Graphcore Ltd.

import argparse
import json
import math
import os
import shutil
import time

import numpy as np

import core.utils as utils
import cv2
import log
import tensorflow as tf
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from ipu_utils import stages_constructor
from log import logger
from tensorflow.python import ipu
from tensorflow.python.ipu import ipu_infeed_queue, ipu_outfeed_queue, loops


class YoloTest(object):
    def __init__(self, opts):
        self.input_size = opts["test"]["input_size"]
        self.classes = utils.read_class_names(opts["yolo"]["classes"])
        self.num_classes = len(self.classes)
        self.score_threshold = opts["test"]["score_threshold"]
        self.iou_threshold = opts["test"]["iou_threshold"]
        self.moving_avg_decay = opts["yolo"]["moving_avg_decay"]
        self.annotation_path = opts["test"]["annot_path"]
        self.weight_file = opts["test"]["weight_file"]
        self.write_image = opts["test"]["write_image"]
        self.write_image_path = opts["test"]["write_image_path"]
        self.show_label = opts["test"]["show_label"]
        self.batch_size = opts["test"]["batch_size"]
        self.precision = tf.float16 if opts["yolo"]["precision"] == "fp16" else tf.float32
        self.use_moving_avg = opts["yolo"]["use_moving_avg"]
        self.repeat_count = opts["test"]["repeat_count"]
        self.use_infeed_queue = opts["test"]["use_infeed_queue"]
        self.predicted_file_path = opts["test"]["predicted_file_path"]
        self.ground_truth_file_path = opts["test"]["ground_truth_file_path"]
        self.meta_dict = {}
        self.testset = Dataset("test", opts)

        # Configure arguments for targeting the IPU
        config = ipu.config.IPUConfig()
        config.auto_select_ipus = 1
        config.configure_ipu_system()

        model = YOLOV3(False, opts)
        # construct model
        # we will put whole network on one ipu
        layers = []
        # build layer functions for backbone and upsample
        layers.extend(model.build_backbone())
        # last layer of darknet53 is classification layer, so it have 52 conv layers
        assert len(layers) == 52
        layers.extend(model.build_upsample())
        # there is 25 conv layers if we count upsmaple as a conv layer
        assert len(layers) == 52+25
        # decoding layer and loss layer is always put on last IPU
        layers.append(model.decode_boxes)

        # reuse stages_constructor so we don't need to pass params by hand
        network_func = stages_constructor(
            [layers],
            ["input_data", "nums"],
            ["pred_sbbox", "pred_mbbox", "pred_lbbox", "nums"])[0]
        input_shape = (self.batch_size, self.input_size, self.input_size, 3)
        self.lines, self.image_dict = self.load_data()
        if self.use_infeed_queue:
            # The dataset for feeding the graphs
            def data_gen():
                return self.data_generator()
            with tf.device("cpu"):
                ds = tf.data.Dataset.from_generator(data_gen,
                                                    output_types=(tf.float16, tf.int32),
                                                    output_shapes=(input_shape, (self.batch_size,))
                                                    )
            ds = ds.repeat()
            ds = ds.prefetch(self.repeat_count*10)
            # The host side queues
            infeed_queue = ipu_infeed_queue.IPUInfeedQueue(ds)
            outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

            def model_func(input_data, nums):
                pred_sbbox, pred_mbbox, pred_lbbox, nums = network_func(input_data, nums)
                outfeed = outfeed_queue.enqueue(
                    {"pred_sbbox": pred_sbbox, "pred_mbbox": pred_mbbox, "pred_lbbox": pred_lbbox, "nums": nums})
                return outfeed

            def my_net():
                r = loops.repeat(self.repeat_count,
                                 model_func, [], infeed_queue)
                return r

            with ipu.scopes.ipu_scope("/device:IPU:0"):
                self.run_loop = ipu.ipu_compiler.compile(
                    my_net, inputs=[])
            # The outfeed dequeue has to happen after the outfeed enqueue
            self.dequeue_outfeed = outfeed_queue.dequeue()
            self.sess = tf.Session(config=tf.ConfigProto())
            self.sess.run(infeed_queue.initializer)
        else:
            # if using feed dict, it will be simpler
            # the cost is throughput
            with tf.device("cpu"):
                with tf.name_scope("input"):
                    # three channel images
                    self.input_data = tf.placeholder(
                        shape=input_shape, dtype=self.precision, name="input_data")
                    self.nums = tf.placeholder(
                        shape=(self.batch_size), dtype=tf.int32, name="nums")

            with ipu.scopes.ipu_scope("/device:IPU:0"):
                self.output = ipu.ipu_compiler.compile(
                    network_func, [self.input_data, self.nums])

            self.sess = tf.Session(
                config=tf.ConfigProto())
        if self.use_moving_avg:
            with tf.name_scope("ema"):
                ema_obj = tf.train.ExponentialMovingAverage(
                    self.moving_avg_decay)
            self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        else:
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weight_file)

    def load_data(self):
        with open(self.annotation_path, "r") as annotation_file:
            # load_all images
            lines = []
            for line in annotation_file:
                lines.append(line)
        image_dict = self.testset.load_images(dump=False)
        return lines, image_dict

    def data_generator(self):
        """Generate input image and write groundtruth info
        """
        if os.path.exists(self.write_image_path):
            shutil.rmtree(self.write_image_path)
        os.mkdir(self.write_image_path)
        self.ground_truth_file = open(self.ground_truth_file_path, "w")

        image_datas = []
        nums = []
        for num, line in enumerate(self.lines):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split("/")[-1]
            image = self.image_dict[line.strip()]
            bbox_data_gt = np.array(
                [list(map(int, box.split(","))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:,
                                                     :4], bbox_data_gt[:, 4]

            num_bbox_gt = len(bboxes_gt)

            # output ground-truth
            self.ground_truth_file.write(str(num)+":\n")
            for i in range(num_bbox_gt):
                class_name = self.classes[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ",".join(
                    [class_name, xmin, ymin, xmax, ymax]) + "\n"
                self.ground_truth_file.write(bbox_mess)
            image_copy = np.copy(image)
            org_h, org_w, _ = image.shape

            image_data = utils.resize_image(
                image_copy, [self.input_size, self.input_size])
            # we don't want to pass metadata through pipeline
            # so we'll keep it with a dictionary
            self.meta_dict[num] = [org_h, org_w, image_name, line]
            image_datas.append(image_data)
            nums.append(num)
            if len(nums) < self.batch_size:
                if num < len(self.lines) - 1:
                    continue
                else:
                    # if there's not enough data to fill the last batch
                    # we repeat the last image to yield a full sized batch
                    for _ in range(len(image_datas), self.batch_size):
                        image_datas.append(image_datas[-1])
                        nums.append(nums[-1])
            image_datas = np.array(image_datas).astype(np.float16)
            yield (image_datas, nums)
            if num < len(self.lines) - 1:
                image_datas = []
                nums = []
        while True:
            # if using infeed_queue. it will need more batches
            # to padd the data and meet the required repeat_count
            # so we will use last batch for padding
            yield (image_datas, nums)

    def parse_result(self, pred_sbbox_list, pred_mbbox_list, pred_lbbox_list, nums):
        """Parse and write predicted result
        """
        for i in range(len(nums)):
            # if nums value is repeated
            # that means nums[i] is a repeated value for matching required batch size
            # so we can stop the iteration
            if i > 0 and nums[i] <= nums[i-1]:
                break
            num = nums[i]
            pred_sbbox = pred_sbbox_list[i]
            pred_mbbox = pred_mbbox_list[i]
            pred_lbbox = pred_lbbox_list[i]
            org_h, org_w, image_name, line = self.meta_dict[num]
            image_path = line.strip().split()[0]
            image = self.image_dict[line.strip()]

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                        np.reshape(
                                            pred_mbbox, (-1, 5 + self.num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
            # convert boxes from input_image coordinate to original image coordinate
            bboxes = utils.postprocess_boxes(
                pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
            bboxes_pr = utils.nms(bboxes, self.iou_threshold)

            if self.write_image:
                image = utils.draw_bbox(
                    image, bboxes_pr, self.classes, show_label=self.show_label)
                cv2.imwrite(self.write_image_path+image_name, image)

            self.predict_result_file.write(str(num)+":\n")
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self.classes[class_ind]
                score = "%.4f" % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ",".join(
                    [class_name, score, xmin, ymin, xmax, ymax]) + "\n"
                self.predict_result_file.write(bbox_mess)

    def evaluate(self):
        self.predict_result_file = open(self.predicted_file_path, "w")
        if self.use_infeed_queue:
            # using infeed queue to improve throughput
            # we can use an additional thread to run dequeue_outfeed for decrease latency and further improve throughput
            total_samples = len(self.lines)
            interaction_samples = self.batch_size*self.repeat_count
            total_interactions = total_samples/interaction_samples
            total_interactions = math.ceil(total_interactions)
            for interaction_index in range(total_interactions):
                run_start = time.time()
                self.sess.run(self.run_loop)
                result = self.sess.run(
                    self.dequeue_outfeed)
                run_duration = time.time()-run_start
                pred_sbbox_list, pred_mbbox_list, pred_lbbox_list, nums = result[
                    "pred_sbbox"], result["pred_mbbox"], result["pred_lbbox"], result["nums"]
                for i in range(len(nums)):
                    # len(nums) == repeat_count
                    # there's repeat count number of batches for each run
                    if i > 0 and nums[i][0] <= nums[i-1][0]:
                        # ignore repeated data
                        # these are only for meeting data size required when using ipu.loops.repeat
                        break
                    self.parse_result(pred_sbbox_list[i], pred_mbbox_list[i], pred_lbbox_list[i], nums[i])
                logger.info("progress:{}/{} ,latency: {}, through put: {}, batch size: {}, repeat count: {}".format(
                    (interaction_index+1)*interaction_samples, len(self.lines),
                    run_duration,
                    interaction_samples/run_duration,
                    self.batch_size,
                    self.repeat_count))
        else:
            # if not use infeed_queue, it will return for every batch
            data_gen = self.data_generator()
            interaction_samples = self.batch_size
            total_interactions = math.ceil(len(self.lines)/interaction_samples)
            for interaction_index in range(total_interactions):
                image_datas, nums = next(data_gen)
                run_start = time.time()
                pred_sbbox_list, pred_mbbox_list, pred_lbbox_list, nums = self.sess.run(
                    self.output,
                    feed_dict={
                        self.input_data: image_datas,
                        self.nums: nums
                    }
                )
                run_duration = time.time()-run_start
                self.parse_result(pred_sbbox_list, pred_mbbox_list, pred_lbbox_list, nums)
                logger.info("progress:{}/{} ,latency: {}, through put: {}, batch size: {}".format(
                    (interaction_index+1)*interaction_samples,
                    len(self.lines),
                    run_duration,
                    interaction_samples/run_duration,
                    self.batch_size))
        self.ground_truth_file.close()
        self.predict_result_file.close()
        self.sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluation in TensorFlow", add_help=False)
    parser.add_argument("--config", type=str, default="config/config_800.json",
                        help="json config file for yolov3.")
    parser.add_argument("--test-annot-path", type=str,
                        help="data path for test")
    parser.add_argument("--weight-file", type=str,
                        help="path for test weights")

    arguments = parser.parse_args()
    with open(arguments.config) as f:
        opts = json.load(f)
    opts['test']['annot_path'] = arguments.test_annot_path or opts['test']['annot_path']
    opts['test']['weight_file'] = arguments.weight_file or opts['test']['weight_file']
    YoloTest(opts).evaluate()
