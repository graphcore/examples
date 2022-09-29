# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import io
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Optional, Tuple
from util.log import logger
import simplejpeg
from PIL import Image
import torch
import poptorch
import time
import json


def get_compile_datum(args, opts, dataset, collate_fn=None):
    loader = poptorch.DataLoader(
        options=opts,
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    loader_iter = iter(loader)
    datum = next(loader_iter)
    return datum


class GeneratedData(object):
    def __init__(self, input_size, use_half, image_transform, pretrain=False):
        self.pretrain = pretrain
        self.image_transform = image_transform
        self.ids_shuffle, self.ids_restore, self.keep_mat, self.restore_mat, self.mask = ImageFolder.generate_mask_patches(
            use_half)
        self.image = transforms.ToPILImage(mode="RGB")(torch.randint(
            low=0, high=255, size=(3, 250, 250), dtype=torch.uint8))
        self.target = 0

    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):
        image = self.image_transform(self.image)
        if self.pretrain:
            return image, self.ids_shuffle, self.ids_restore, self.keep_mat, self.restore_mat, self.mask
        else:
            return image, self.target


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if not args.generated_data:
        root = os.path.join(args.data_path,
                            'train' if is_train else 'validation')
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        dataset = GeneratedData(
            args.input_size,
            args.half,
            transform,
            args.pretrain)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        # to maintain same ratio w.r.t. 224 images
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp")


class ImageFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        use_half=True,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.use_half = use_half
        self.is_initialized = False
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.read_samples_with_cache(
            class_to_idx, IMG_EXTENSIONS, is_valid_file)

    def read_samples_with_cache(self, class_to_idx, extensions, is_valid_file):
        start_write_time = time.time()
        samples = self.make_dataset(
            self.root, class_to_idx, extensions, is_valid_file)
        Scan_samples_time = time.time() - start_write_time
        logger.info(f'Scan samples time: {Scan_samples_time}')
        logger.info(f'Writing cache...')
        with open("./dataset_cache", 'w') as f:
            json.dump(samples, f)
        logger.info(f'Writing done')
        return samples

    @staticmethod
    def generate_mask_patches(use_half):

        L, D, mask_ratio = 196, 768, 0.75
        noise = torch.rand(L)  # noise in [0, 1]
        # sort noise for each sampleÅÅÅ
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        len_keep = int(L * (1 - mask_ratio))
        keep_mat = torch.zeros(L, len_keep)
        ids_keep = ids_shuffle[:len_keep]
        keep_mat.scatter_(dim=0, index=ids_keep.view(1, len_keep), value=1)
        restore_mat = torch.zeros(L, L)

        restore_mat.scatter_(dim=0, index=ids_restore.view(1, L), value=1)

        mask = torch.ones([L])
        mask[:len_keep] = 0
        mask = torch.gather(mask, dim=0, index=ids_restore)
        if use_half:
            keep_mat = keep_mat.half()
            restore_mat = restore_mat.half()

        return ids_shuffle, ids_restore, keep_mat, restore_mat, mask

    def real_init(self):
        classes, class_to_idx = self.find_classes(self.root)
        start_load_time = time.time()
        f = open("./dataset_cache", 'r')
        samples = json.load(f)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.is_initialized = True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.is_initialized is False:
            self.real_init()
        path, target = self.samples[index]

        with open(path, 'rb') as jpeg_file:
            img = jpeg_file.read()
        try:
            image = Image.fromarray(
                simplejpeg.decode_jpeg(
                    img, colorspace='RGB'))
        except BaseException:
            image = Image.open(io.BytesIO(img))
            image = image.convert("RGB")

        ids_shuffle, ids_restore, keep_mat, restore_mat, mask = self.generate_mask_patches(
            self.use_half)
        return self.transform(
            image), ids_shuffle, ids_restore, keep_mat, restore_mat, mask
