# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import io
import logging
import pathlib
import turbojpeg
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from ctypes import *


class ExtendedTurboJPEG(turbojpeg.TurboJPEG):
    def __init__(self, lib_path=None):
        if lib_path is None:
            root_folder = str(pathlib.Path(__file__).parent.resolve())
            lib_path = root_folder + "/turbojpeg/lib/libturbojpeg.so"
        try:
            super().__init__(lib_path)
            turbo_jpeg = cdll.LoadLibrary(
                self.__find_turbojpeg() if lib_path is None else lib_path)
            self.__decompress_crop = turbo_jpeg.tjCropDecompress2
        except OSError:
            logging.warning("TurboJPEG not found, fallback to slower PIL image loading.")


    def _crop_decode_binding(self, img_buf, crop_x, crop_y, crop_h, crop_w, pixel_format=turbojpeg.TJPF_RGB,
                             flags=turbojpeg.TJFLAG_FASTUPSAMPLE | turbojpeg.TJFLAG_FASTDCT):
        try:
            handle = self._TurboJPEG__init_decompress()
            jpeg_array = np.frombuffer(img_buf, dtype=np.uint8)
            src_addr = self._TurboJPEG__getaddr(jpeg_array)
            img_array = np.empty(
                [crop_h + 8, crop_w, turbojpeg.tjPixelSize[pixel_format]],
                dtype=np.uint8)  # one additional block row to avoid segment error
            img_array = img_array[:crop_h, :crop_w, :]  # remove added block row
            dest_addr = self._TurboJPEG__getaddr(img_array)
            status = self.__decompress_crop(
                handle, src_addr, jpeg_array.size, dest_addr, 0,
                0, 0, pixel_format, flags, crop_x, crop_y, crop_h, crop_w)
            if status != 0:
                self._TurboJPEG__report_error(handle)
            return img_array
        finally:
            self._TurboJPEG__destroy(handle)


    def crop_decode(self, img_buf, crop_x, crop_y, crop_h, crop_w, pixel_format=turbojpeg.TJPF_RGB):
        try:
            img = self._crop_decode_binding(img_buf, crop_x, crop_y, crop_h, crop_w, pixel_format)
            img = Image.fromarray(img)
            return img
        except BaseException:
            img = Image.open(io.BytesIO(img_buf))
            img = img.convert("RGB")
            img = transforms.functional.crop(img, crop_x, crop_y, crop_w, crop_h)
            return img


    def align_crop(self, crop_x, crop_y, crop_h, crop_w, jpeg_subsample):
        """
        Align the crop coordinates to the block borders.
        """
        crop_y -= crop_y % turbojpeg.tjMCUWidth[jpeg_subsample]  # align to block
        crop_w -= crop_w % turbojpeg.tjMCUWidth[jpeg_subsample]
        crop_w = max(crop_w, turbojpeg.tjMCUWidth[jpeg_subsample])  # make sure the numbers are positive
        crop_x -= crop_x % turbojpeg.tjMCUHeight[jpeg_subsample]
        crop_h -= crop_h % turbojpeg.tjMCUHeight[jpeg_subsample]
        crop_h = max(crop_h, turbojpeg.tjMCUHeight[jpeg_subsample])
        return crop_x, crop_y, crop_h, crop_w
