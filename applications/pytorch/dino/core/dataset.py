# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import torch
from torch.utils.data import dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter, ImageOps
import popdist
import numpy as np


class To_Tensor(torch.nn.Module):
    def forward(self, img):
        out = torch.from_numpy(np.array(img))
        out = out.permute(2, 0, 1)
        return out


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationDINO(object):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.local_crops_number = local_crops_number

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                To_Tensor(),
            ])
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                To_Tensor(),
            ])
        # transformation for the local small crops
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96,
                    scale=local_crops_scale,
                    interpolation=InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(
                    p=0.5),
                To_Tensor(),
            ])

    def __call__(self, image):
        global_img = []
        crops = []
        gl1 = self.global_transfo1(image)
        gl2 = self.global_transfo2(image)
        global_img.append(gl1.unsqueeze(0))
        global_img.append(gl2.unsqueeze(0))
        for _ in range(self.local_crops_number):
            loc = self.local_transfo(image)
            crops.append(loc.unsqueeze(0))
        return torch.cat(global_img), torch.cat(crops)


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        return self.transform(image)
