# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# --------------------------------------------------------
# Swin Transformer
# This file has been modified by Graphcore Ltd.
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# The LICENSE referenced above is reproduced below:
# MIT License
#
#     Copyright (c) Microsoft Corporation.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE
# Written by Ze Liu
# --------------------------------------------------------
import popdist
import os
import torch
import numpy as np
import poptorch
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .ipu_mixup import Mixup
from timm.data import create_transform
from .raw_imagenet import ImageFolder
from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from PIL import Image

from .preprocess import *

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except BaseException:
    from timm.data.transforms import _pil_interp


class collater():
    def __init__(self, config, mixup_fn=None):
        self.mixup_fn = mixup_fn
        self.config = config

    def __call__(self, batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        data = torch.stack(data)
        target = torch.tensor(target)
        if data.shape[0] % 2 == 0:
            data, targets = self.mixup_fn(data, target)
        else:

            print("WARNING: Batchsize is not even! ")
            # fix data shape for uncomplete batch when rebatch is enabled
            if data.shape[0] == 1:
                data, targets = self.mixup_fn(data.repeat(2, 1, 1, 1), target.repeat(2))
            else:
                data, targets = self.mixup_fn(data[0:-1, :, :, :], target[0:-1])
        if self.config.PRECISION[0] == 'half':
            data = data.half()
        return [data, targets]


def build_loader(config, opts):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config)
    config.freeze()
    print(f"Data loaded with {len(dataset_train)} train  imgs.")

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES)

    collate_fn = collater(config, mixup_fn)
    data_loader_train = poptorch.DataLoader(
        options=opts,
        dataset=dataset_train,
        shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        mode=poptorch.DataLoaderMode.AsyncRebatched,
        async_options={'early_preload': True,
                       "miss_sleep_time_in_ms": 0,
                       "buffer_size": 4},
        persistent_workers=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        rebatched_worker_size=1024,
        collate_fn=collate_fn
    )
    return dataset_train, data_loader_train, mixup_fn


def build_dataloader_val(config, opts):
    dataset_val, _ = build_dataset(is_train=False, config=config)
    config.freeze()
    print(f"Data loaded with {len(dataset_val)} train  imgs.")
    data_loader_val = poptorch.DataLoader(
        options=opts,
        dataset=dataset_val,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        mode=poptorch.DataLoaderMode.Async,
        async_options={'early_preload': True, "miss_sleep_time_in_ms": 0},
        persistent_workers=True,
        drop_last=True
    )
    return dataset_val, data_loader_val


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'validation'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(
                config.DATA.DATA_PATH,
                ann_file,
                prefix,
                transform,
                cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = ImageFolder(root, transform=transform)




        nb_classes = config.MODEL.NUM_CLASSES
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE[0] > 32

    input_size = config.DATA.IMG_SIZE[0]
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        added_transforms = [IgnoreBboxIfPresent(), LoadJpeg()]
        transform = create_transform(
            input_size=input_size,  # config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4)
        return transforms.Compose(added_transforms + transform.transforms)

    t = []
    t.append(IgnoreBboxIfPresent())
    t.append(LoadJpeg())
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE[0])
            t.append(
                transforms.Resize(
                    size, interpolation=_pil_interp(
                        config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE[0],
                     config.DATA.IMG_SIZE[1]),
                    interpolation=_pil_interp(
                        config.DATA.INTERPOLATION)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
