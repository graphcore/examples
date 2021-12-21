# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2021 RangiLyu.
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
# This file has been modified by Graphcore Ltd.

import numpy as np
import torch
from .coco import CocoDataset
from pycocotools.coco import COCO
from utils import logger


if logger.GLOBAL_LOGGER is not None:
    print = logger.GLOBAL_LOGGER.log_str


def calc_area(boxes):
    # boxes: n, 4
    # return: n
    x1, y1, x2, y2 = np.split(boxes, 4, 1)
    areas = (y2 - y1) * (x2 - x1)  # n,1
    return areas[:, 0]


class CocoDatasetForRcnn(CocoDataset):
    def __init__(self,
                 preset_indices=None,
                 area_filter_thrd=0.0,
                 num_gtboxes=20,
                 specified_length=None,
                 extra_layer=None,
                 **kwargs):
        self.area_filter_thrd = area_filter_thrd
        self.num_gtboxes = num_gtboxes
        self.preset_indices = preset_indices
        self._cur_for_preset_indices = 0
        super(CocoDataset, self).__init__(**kwargs)
        self.real_length = len(self.data_info)
        self.length = self.real_length if specified_length is None else specified_length
        self.extra_layer = extra_layer

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
            'file_name': '000000000139.jpg',
            'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
            'height': 426,
            'width': 640,
            'date_captured': '2013-11-21 01:34:01',
            'flickr_url':
                'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
            'id': 139},
            ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i+1 for i, cat_id in enumerate(self.cat_ids)}
        print('self.cat_ids', self.cat_ids)
        print('self.cat2label', self.cat2label)
        self.cats = self.coco_api.loadCats(self.cat_ids)
        print('self.cats', self.cats)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        filter zero area boxes
        """
        if self.preset_indices is None:
            pass
        else:
            idx = self.preset_indices[self._cur_for_preset_indices]
            self._cur_for_preset_indices += 1

        idx = int(idx % self.real_length)
        meta = super().get_train_data(idx)

        # filter boxes and labels by area
        areas = calc_area(meta['gt_bboxes'])
        mask = areas > self.area_filter_thrd
        meta['gt_bboxes'] = meta['gt_bboxes'][mask, :]
        meta['gt_labels'] = meta['gt_labels'][mask]
        meta['db_inds'] = idx

        # pad boxes and inds
        boxes = np.zeros((self.num_gtboxes, 4)).astype(np.float32)
        num_boxes = meta['gt_bboxes'].shape[0]
        boxes[:num_boxes, :] = meta['gt_bboxes'][:self.num_gtboxes]
        meta['gt_bboxes'] = torch.from_numpy(boxes)
        labels = np.asarray([0] * self.num_gtboxes)
        labels[:num_boxes] = meta['gt_labels'][:self.num_gtboxes]
        meta['gt_labels'] = torch.from_numpy(labels)
        meta['num_boxes'] = num_boxes

        if num_boxes == 0:
            return None  # return None will re-run this function

        # proc data in extra layer
        if self.extra_layer is not None:
            meta = self.extra_layer(meta)

        return meta

    def __len__(self):
        return self.length
