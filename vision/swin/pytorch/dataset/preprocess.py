# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import io
import math
import torch
from torchvision import transforms
from PIL import Image
import simplejpeg


class IgnoreBboxIfPresent(torch.nn.Module):
    def forward(self, img):
        if isinstance(img, tuple):
            return img[0]
        return img


class LoadJpeg(torch.nn.Module):
    def forward(self, img):
        if isinstance(
                img,
                Image.Image) or isinstance(
                img,
                type(Image)) or isinstance(
                img,
                torch.Tensor):
            return img
        else:
            try:

                img_array = simplejpeg.decode_jpeg(img, colorspace='RGB')

                return Image.fromarray(img_array)
            except BaseException:
                # fallback to PIL if SimpleJPEG unavailable or jpeg encode not
                # supported
                img = Image.open(io.BytesIO(img))
                img = img.convert("RGB")
                return img
