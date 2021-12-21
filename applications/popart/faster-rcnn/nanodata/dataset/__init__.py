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

import copy
import warnings

from .coco import CocoDataset
from .xml_dataset import XMLDataset
from .xml_dataset_for_rcnn import XMLDatasetForRcnn
from .coco_dataset_for_rcnn import CocoDatasetForRcnn


def build_dataset(cfg, mode, preset_indices=None, specified_length=None, extra_layer=None):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")
    if name == "coco":
        assert preset_indices is None
        warnings.warn(
            "Dataset name coco has been deprecated. Please use CocoDataset instead."
        )
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "xml_dataset":
        assert preset_indices is None
        warnings.warn("Dataset name xml_dataset has been deprecated. "
                      "Please use XMLDataset instead.")
        return XMLDataset(mode=mode, **dataset_cfg)
    elif name == "CocoDataset":
        assert preset_indices is None
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "XMLDataset":
        assert preset_indices is None
        return XMLDataset(mode=mode, **dataset_cfg)
    elif name == "XMLDatasetForRcnn":
        # filter bboxes which's area less than {area_filter_thrd}
        assert 'area_filter_thrd' in dataset_cfg
        return XMLDatasetForRcnn(mode=mode,
                                 preset_indices=preset_indices,
                                 specified_length=specified_length,
                                 extra_layer=extra_layer,
                                 **dataset_cfg)
    elif name == "coco_forRcnn":
        # filter bboxes which's area less than {area_filter_thrd}
        assert 'area_filter_thrd' in dataset_cfg
        dataset_cfg.pop('include_difficult')
        dataset_cfg.pop('class_names')
        return CocoDatasetForRcnn(mode=mode,
                                  preset_indices=preset_indices,
                                  specified_length=specified_length,
                                  extra_layer=extra_layer,
                                  **dataset_cfg)
    else:
        raise NotImplementedError("Unknown dataset type!")
