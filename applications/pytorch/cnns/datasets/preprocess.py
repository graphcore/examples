# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
from torchvision import transforms


normalization_parameters = {"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]}

use_bbox_info_config = {False: {"max_trial": 1, "minimum_bbox_interlap": 0.0},
                        True: {"max_trial": 10, "minimum_bbox_interlap": 0.1}}


def get_preprocessing_pipeline(train, input_size=224, half_precision=False, normalize=True, eightbit=False, use_bbox_info=False):
    """
    Return optimized pipeline, which contains fused transformations.
    """
    pipeline_steps = []
    if train:
        pipeline_steps.append(RandomResizedFlipCrop(input_size, **use_bbox_info_config[use_bbox_info]))
    else:
        # 'resize_size' is scaled by the specified 'input_size' to allow for arbitrary-sized images.
        resize_size = int(input_size * 256.0 / 224.0)
        pipeline_steps = [transforms.Resize(resize_size), transforms.CenterCrop(input_size)]

    if normalize:
        pipeline_steps.append(NormalizeToTensor(mean=normalization_parameters["mean"], std=normalization_parameters["std"]))
    else:
        # Return tensor
        pipeline_steps.append(NormalizeToTensor.pil_to_tensor)
        # if not normalized full precision, need to be converted to float.
        if not half_precision:
            pipeline_steps.append(ToFloat())

    if eightbit:
        pipeline_steps.append(ToByte())
    elif half_precision:
        pipeline_steps.append(ToHalf())

    return transforms.Compose(pipeline_steps)


class ToHalf(torch.nn.Module):
    def forward(self, tensor):
        return tensor.half()


class ToFloat(torch.nn.Module):
    def forward(self, tensor):
        return tensor.float()


class ToByte(torch.nn.Module):
    def forward(self, tensor):
        return tensor.byte()


class RandomResizedFlipCrop(transforms.RandomResizedCrop):
    """
    Fuse RandomResizedCrop and RandomHorizontalFlip augmentation.
    The horizontal flip happens before the resize, depends on the croped imge size
    """
    def __init__(self, *args, max_trial=1, minimum_bbox_interlap=0.0, **kwargs):
        self.max_trial = max_trial
        self.minimum_bbox_interlap = minimum_bbox_interlap
        super(RandomResizedFlipCrop, self).__init__(*args, **kwargs)

    def get_bbox(self, img, bbox=None):
        if bbox is None:
            return self.get_params(img, self.scale, self.ratio)
        trial_nr = 1
        # adjust bbox with image sizes
        w, h = transforms.functional._get_image_size(img)
        bbox = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        while trial_nr < self.max_trial:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            dx = min(i + h, bbox[2]) - max(i, bbox[0])
            dy = min(j + w, bbox[3]) - max(j, bbox[1])
            if h * w * self.minimum_bbox_interlap <= dx * dy:
                return i, j, h, w
            trial_nr += 1
        return i, j, h, w


    def __call__(self, img):
        if isinstance(img, tuple):   # unpack bbox values if available
            bbox = img[1]
            img = img[0]
        else:
            bbox = None
        i, j, h, w = self.get_bbox(img, bbox)
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
