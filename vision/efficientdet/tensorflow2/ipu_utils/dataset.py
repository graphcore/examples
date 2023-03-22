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

import argparse
from functools import reduce
from pathlib import Path
from typing import List, Text, Tuple, Union

import numpy as np
import tensorflow as tf
import utils
from ipu_automl_io import preprocess_resize
from PIL import Image


T_DatasetScaleRawImages = Tuple[tf.data.Dataset, int, tf.Tensor]


def global_batch_size(args: argparse.Namespace) -> int:
    return args.micro_batch_size


def input_tensor_shape(
    args: argparse.Namespace, image_size: Union[Text, int, Tuple[int, int]], num_channels: int = 3
) -> Tuple[int, int]:
    return utils.parse_image_size(image_size) + (num_channels,)


def _configure_dataset(dataset: tf.data.Dataset, args: argparse.Namespace, dataset_repeats: int = 1) -> tf.data.Dataset:
    dataset = dataset.map(lambda x: tf.cast(x, dtype=args.io_precision))
    dataset = dataset.cache()

    # Repeat the image "benchmark_repeats" times
    dataset = dataset.repeat(dataset_repeats)
    dataset = dataset.batch(args.micro_batch_size, drop_remainder=True)

    # Now repeat the entire dataset "num_repeats" times
    dataset = dataset.repeat(args.num_repeats)
    if args.dataset_prefetch_buffer > 0:
        dataset = dataset.prefetch(args.dataset_prefetch_buffer)
    return dataset


def generated_inference_dataset(
    args: argparse.Namespace,
    image_size: Union[Text, int, Tuple[int, int]],
) -> T_DatasetScaleRawImages:
    num_samples = global_batch_size(args)

    inputs = tf.random.uniform((num_samples,) + input_tensor_shape(args, image_size))

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return _configure_dataset(dataset, args, args.benchmark_repeats), 1, inputs


def repeated_image_dataset(
    args: argparse.Namespace, image_size: Union[Text, int, Tuple[int, int]]
) -> T_DatasetScaleRawImages:
    num_samples = global_batch_size(args)
    imgs = [np.array(Image.open(args.image_path))] * num_samples
    imgs = tf.convert_to_tensor(imgs)

    print("Preprocessing")
    inputs, scales = preprocess_resize(imgs, image_size)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return _configure_dataset(dataset, args, args.benchmark_repeats), scales, imgs


def image_directory_dataset(
    args: argparse.Namespace, image_size: Union[Text, int, Tuple[int, int]]
) -> T_DatasetScaleRawImages:

    img_height, img_width = utils.parse_image_size(image_size)
    extensions = ["jpg", "png", "jpeg"]

    def load_imgs_from_ext(acc: List[Text], ext: Text) -> List[Text]:
        imgs = [p for p in Path(args.image_path).rglob("*." + ext)]
        return acc + imgs

    img_paths = reduce(load_imgs_from_ext, extensions, [])
    imgs = [tf.convert_to_tensor(Image.open(p)) for p in img_paths]

    print("Preprocessing")
    inputs, scales = preprocess_resize(imgs, image_size)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return _configure_dataset(dataset, args), scales, imgs


def get_dataset(args: argparse.Namespace, image_size: Union[Text, int, Tuple[int, int]]) -> T_DatasetScaleRawImages:
    if args.dataset_type == "repeated-image" or args.dataset_type == "single-image":
        return repeated_image_dataset(args, image_size)
    elif args.dataset_type == "generated":
        return generated_inference_dataset(args, image_size)
    elif args.dataset_type == "image-directory":
        return image_directory_dataset(args, image_size)
    else:
        raise NotImplementedError(f"Dataset type {args.dataset_type} has not been implemented")
