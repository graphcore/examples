# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
from torchvision import transforms


normalization_parameters = {"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]}


def get_preprocessing_pipeline(train, input_size=224, half_precision=False, normalize=True):
    """
    Return optimized pipeline, which contains fused transformations.
    """
    pipeline_steps = [transforms.Resize(256)]
    if train:
        pipeline_steps.append(RandomResizedFlipCrop(input_size))
    else:
        pipeline_steps.append(transforms.CenterCrop(input_size))

    if normalize:
        pipeline_steps.append(NormalizeToTensor(mean=normalization_parameters["mean"], std=normalization_parameters["std"]))
    else:
        # Return tensor
        pipeline_steps.append(NormalizeToTensor.pil_to_tensor)
        # if not normalized full precision, need to be converted to float.
        if not half_precision:
            pipeline_steps.append(ToFloat())

    if half_precision:
        pipeline_steps.append(ToHalf())
    return transforms.Compose(pipeline_steps)


class ToHalf(torch.nn.Module):
    def forward(self, tensor):
        return tensor.half()


class ToFloat(torch.nn.Module):
    def forward(self, tensor):
        return tensor.float()


class RandomResizedFlipCrop(transforms.RandomResizedCrop):
    """
    Fuse RandomResizedCrop and RandomHorizontalFlip augmentation.
    """
    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(img, torch.Tensor):
            tensor = torch.unsqueeze(img, 0)
            tensor = transforms.functional_tensor.crop(tensor, i, j, h, w)
            if torch.rand(1) < 0.5:
                tensor = self.fast_hflip(tensor)
            tensor = tensor.float()
            tensor = torch.nn.functional.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)
            return tensor.squeeze(0)
        else:
            img = transforms.functional.crop(img, i, j, h, w)
            if torch.rand(1) < 0.5:
                img = transforms.functional_pil.hflip(img)
            img = transforms.functional.resize(img, self.size, self.interpolation)
            return img

    @staticmethod
    def fast_hflip(x):
        reverse_index = torch.arange(x.size()[-1] - 1, -1, -1)
        return x[:, :, :, reverse_index]


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
