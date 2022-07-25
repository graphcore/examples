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
from .xml_dataset import XMLDataset
from utils import logger


if logger.GLOBAL_LOGGER is not None:
    print = logger.GLOBAL_LOGGER.log_str


def calc_area(boxes):
    # boxes: n,4
    # return
    x1, y1, x2, y2 = np.split(boxes, 4, 1)
    areas = (y2 - y1) * (x2 - x1)  # n,1
    return areas[:, 0]


class XMLDatasetForRcnn(XMLDataset):
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
        super(XMLDatasetForRcnn, self).__init__(**kwargs)
        self.real_length = len(self.data_info)
        self.length = self.real_length * 2 if specified_length is None else specified_length
        self.extra_layer = extra_layer

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
        #

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
