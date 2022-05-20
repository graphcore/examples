# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import io
import math
import torch
from torchvision import transforms
from PIL import Image
import turbojpeg
from datasets.optimised_jpeg import ExtendedTurboJPEG


_jpeg_decoder = ExtendedTurboJPEG()

normalization_parameters = {"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]}

use_bbox_info_config = {False: {"max_trial": 1, "minimum_bbox_interlap": 0.0},
                        True: {"max_trial": 10, "minimum_bbox_interlap": 0.1}}


def get_preprocessing_pipeline(train, input_size=224, half_precision=False, normalize=True, eightbit=False, use_bbox_info=False, fine_tuning=False):
    """
    Return optimized pipeline, which contains fused transformations.
    """
    pipeline_steps = []
    if train and not fine_tuning:
        pipeline_steps += [RandomResizedBoxCrop(input_size, **use_bbox_info_config[use_bbox_info]), transforms.RandomHorizontalFlip()]
    else:
        # 'resize_size' is scaled by the specified 'input_size' to allow for arbitrary-sized images.
        resize_size = int(input_size * 256.0 / 224.0)
        pipeline_steps += [IgnoreBboxIfPresent(), LoadJpeg(), transforms.Resize(resize_size), transforms.CenterCrop(input_size)]

    if normalize:
        pipeline_steps.append(NormalizeToTensor(mean=normalization_parameters["mean"], std=normalization_parameters["std"]))
    else:
        pipeline_steps.append(NormalizeToTensor.pil_to_tensor)

    if eightbit:
        pipeline_steps.append(ToByte())
    elif half_precision:
        pipeline_steps.append(ToHalf())
    elif not normalize:
        pipeline_steps.append(ToFloat())  # no normalisation results byte tensor -> must be converted to float

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


class RandomResizedBoxCrop(transforms.RandomResizedCrop):
    """
    RandomResized considering bounding boxes.
    """
    def __init__(self, *args, max_trial=1, minimum_bbox_interlap=0.0, **kwargs):
        self.max_trial = max_trial
        self.minimum_bbox_interlap = minimum_bbox_interlap
        super(RandomResizedBoxCrop, self).__init__(*args, **kwargs)

    def get_bbox(self, img, bbox=None):
        if bbox is None:
            return self.get_params(img, self.scale, self.ratio)
        trial_nr = 1
        # adjust bbox with image sizes
        w, h = transforms.functional.get_image_size(img)
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
        if isinstance(img, type(Image)) or isinstance(img, Image.Image):
            img = self.pil_augment(img, bbox)   # PIL Image augmentation
        else:
            img = self.jpeg_augment(img, bbox)  # JPEG stream augmentation
        img = transforms.functional.resize(img, self.size, self.interpolation)
        return img

    def pil_augment(self, img, bbox):
        i, j, h, w = self.get_bbox(img, bbox)
        img = transforms.functional.crop(img, i, j, h, w)
        return img

    def jpeg_augment(self, img, bbox):
        try:
            width, height, jpeg_subsample, _ = _jpeg_decoder.decode_header(img)
            fake_image = Image.new(size=(width, height), mode='L')  # create a fake image
            i, j, h, w = self.get_bbox(fake_image, bbox)
            i, j, h, w = _jpeg_decoder.align_crop(i, j, h, w, jpeg_subsample)  # align crop to block size
            img = _jpeg_decoder.crop_decode(img, i, j, h, w)
        except BaseException:
            # fallback to PIL if TurboJPEG unavailable or jpeg encode not supported
            img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            img = self.pil_augment(img, bbox)
        return img


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
        img = self.pil_to_tensor(img).float()
        img.mul_(self.mul)
        img.sub_(self.sub)
        return img

    @staticmethod
    def pil_to_tensor(pic):
        if isinstance(pic, torch.Tensor):
            return pic
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        return img


class IgnoreBboxIfPresent(torch.nn.Module):
    def forward(self, img):
        if isinstance(img, tuple):
            return img[0]
        return img


class LoadJpeg(torch.nn.Module):
    def forward(self, img):
        if isinstance(img, Image.Image) or isinstance(img, type(Image)) or isinstance(img, torch.Tensor):
            return img
        else:
            try:
                img_array = _jpeg_decoder.decode(img, pixel_format = turbojpeg.TJPF_RGB,
                                                 flags=turbojpeg.TJFLAG_FASTUPSAMPLE | turbojpeg.TJFLAG_FASTDCT)
                return Image.fromarray(img_array)
            except BaseException:
                # fallback to PIL if TurboJPEG unavailable or jpeg encode not supported
                img = Image.open(io.BytesIO(img))
                img = img.convert("RGB")
                return img
