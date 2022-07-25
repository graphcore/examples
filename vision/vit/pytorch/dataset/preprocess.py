# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import torch
from torchvision import transforms
from dataset.customized_randaugment import ImageNetPolicy


normalization_parameters = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def get_preprocessing_pipeline(train, input_size=224, half_precision=False, normalize=True, extra_aug=None):
    """
    Return optimized pipeline, which contains fused transformations.
    """
    pipeline_steps = []
    if train:
        pipeline_steps.append(transforms.RandomResizedCrop((input_size, input_size),
                                                           scale=(0.05, 1.0)))
        if extra_aug is None:
            pass
        elif extra_aug == "imagenet_policy":
            pipeline_steps.append(ImageNetPolicy())
        else:
            raise NotImplementedError(
                "Cifar-10, Cifar-100 policies not implemented.")
    else:
        pipeline_steps.append(transforms.Resize((256)))
        pipeline_steps.append(transforms.CenterCrop(input_size))

    pipeline_steps.append(transforms.ToTensor())
    if normalize:
        pipeline_steps.append(transforms.Normalize(mean=normalization_parameters["mean"],
                                                   std=normalization_parameters["std"]))
    else:
        # Return tensor
        pipeline_steps.append(NormalizeToTensor.pil_to_tensor)

    if half_precision:
        pipeline_steps.append(ToHalf())
    else:
        pipeline_steps.append(ToFloat())

    return transforms.Compose(pipeline_steps)


class ToHalf(torch.nn.Module):
    def forward(self, tensor):
        return tensor.half()


class ToFloat(torch.nn.Module):
    def forward(self, tensor):
        return tensor.float()


class NormalizeToTensor(torch.nn.Module):
    def __init__(self, mean, std):
        """
        Fuse ToTensor and Normalize operation.
        Expected input is a PIL image and the output is the normalized float tensor.
        """
        # fuse: division by 255 and the normalization
        # Convert division to multiply
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.mul = (1.0/(255.0 * std)).view(-1, 1, 1)
        self.sub = (mean / std).view(-1, 1, 1)
        super().__init__()

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = self.pil_to_tensor(img).float()
        if not img.dtype == torch.float:
            img = img.float()
        img.mul_(self.mul)
        img.sub_(self.sub)
        return img

    @staticmethod
    def pil_to_tensor(pic):
        # If it is already tensor, return it.
        if isinstance(pic, torch.Tensor):
            return pic
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        return img
