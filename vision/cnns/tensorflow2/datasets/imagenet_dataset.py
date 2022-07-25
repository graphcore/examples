# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
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
# This file has been modified by Graphcore Ltd.

from . import abstract_dataset
from typing import Callable, Optional
import tensorflow as tf
import os
import glob
import logging
import popdist
from custom_exceptions import DimensionError
from . import image_normalization
from . import application_dataset

IMAGENET_DS_SIZE = {'train': 1281167, 'test': 50000, 'validation': 50000}

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NORMALISATION_MEAN = [0.485, 0.456, 0.406]
NORMALISATION_STD = [0.229, 0.224, 0.225]


def tfrecord_prefix_from_split(split: str) -> str:
    return 'train' if 'train' in split else 'validation'


def expected_num_files_with_prefix(tfrecord_prefix: str) -> int:
    return 1024 if tfrecord_prefix == 'train' else 128


class ImagenetDataset(abstract_dataset.AbstractDataset):

    logger = logging.getLogger('imagenet_dataset')

    def __init__(self,
                 dataset_path: str,
                 split: str,
                 shuffle: bool = True,
                 deterministic: bool = False,
                 seed: Optional[int] = None,
                 img_datatype: tf.dtypes.DType = tf.float32,
                 accelerator_side_preprocess: bool = False,
                 fused_preprocessing: bool = False):

        # The path is the one of dataset under TFRecord format
        if not os.path.exists(dataset_path):
            raise NameError(f'Directory {dataset_path} does not exist')

        if fused_preprocessing is True and accelerator_side_preprocess is False:
            raise ValueError('Fused preprocessing can only be done on the IPU. '
                             'Please enable preprocessing on the IPU.')

        self.dataset_path = dataset_path
        self.split = split
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.seed = seed
        self.img_datatype = img_datatype
        self.accelerator_side_preprocess = accelerator_side_preprocess
        self.fused_preprocessing = fused_preprocessing
        self.cycle_length = 4 if not deterministic else 1
        self.block_length = 4 if not deterministic else 1
        self.shuffle_buffer = 10000

    def read_single_image(self) -> application_dataset.ApplicationDataset:

        tfrecord_prefix = tfrecord_prefix_from_split(self.split)

        filenames = glob.glob1(self.dataset_path, f'{tfrecord_prefix}*')

        num_files = len(filenames)
        expected_num_files = expected_num_files_with_prefix(tfrecord_prefix)
        if num_files != expected_num_files:
            raise ValueError(f'{self.split} dataset should contain {expected_num_files} '
                             f'files but it contains {num_files} instead')

        filenames = list(
            map(lambda filename: os.path.join(self.dataset_path, filename), filenames))
        ImagenetDataset.logger.debug(f'filenames = {filenames}')
        ds = tf.data.Dataset.from_tensor_slices(filenames)

        if popdist.getNumInstances() > 1:
            ds = ds.shard(num_shards=popdist.getNumInstances(), index=popdist.getInstanceIndex())

        if self.split == 'train' and self.shuffle:
            # Shuffle the input files
            ds = ds.shuffle(buffer_size=num_files // popdist.getNumInstances(), seed=self.seed)

        ds = ds.interleave(tf.data.TFRecordDataset,
                           cycle_length=self.cycle_length,
                           block_length=self.block_length,
                           num_parallel_calls=self.cycle_length,
                           deterministic=self.deterministic)

        ImagenetDataset.logger.info(f'dataset = {ds}')

        num_examples = IMAGENET_DS_SIZE[self.split]

        ImagenetDataset.logger.info(f'number of examples {num_examples}')

        iterator = iter(ds)
        first_elem = iterator.get_next()

        image, _ = parse_imagenet_record(
            first_elem, True, tf.float32, seed=self.seed)

        if len(image.shape) != 3:
            raise DimensionError(
                'Dataset input image should have at least 3 dimensions (h,w,c) '
                f'but it has {len(first_elem[0].shape)}')

        num_classes = 1000

        ds = ds.cache()

        return application_dataset.ApplicationDataset(pipeline=ds,
                                                      size=num_examples,
                                                      image_shape=image.shape,
                                                      num_classes=num_classes)

    def cpu_preprocessing_fn(self) -> Callable:

        if self.accelerator_side_preprocess:
            cpu_preprocess_fn = None
        else:
            cpu_preprocess_fn = _imagenet_normalize

        def processing_fn(raw_record): return parse_imagenet_record(
            raw_record, self.split == 'train', self.img_datatype, cpu_preprocess_fn, self.seed)

        return processing_fn

    def ipu_preprocessing_fn(self) -> Callable:

        if self.accelerator_side_preprocess is False:
            return None

        if self.fused_preprocessing:
            preprocessing_fn = _imagenet_fused_normalize
        else:
            preprocessing_fn = _imagenet_normalize

        return preprocessing_fn

    def post_preprocessing_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        if self.split == 'train' and self.shuffle:
            ds = ds.shuffle(self.shuffle_buffer, seed=self.seed)
        return ds


def _imagenet_normalize(image):
    return image_normalization.image_normalisation(image,
                                                   NORMALISATION_MEAN,
                                                   NORMALISATION_STD)


def _imagenet_fused_normalize(image):
    return image_normalization.fused_image_normalisation(image,
                                                         NORMALISATION_MEAN,
                                                         NORMALISATION_STD)


def parse_imagenet_record(raw_record, is_training, dtype, cpu_preprocess_fn=None, seed=None):
    """Parses a record containing a training example of an image.
    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).
    Args:
      raw_record: scalar Tensor tf.string containing a serialized Example protocol
        buffer.
      is_training: A boolean denoting whether the input is for training.
      dtype: data type to use for images/features.
    Returns:
      Tuple with processed image tensor in a channel-last format and
      one-hot-encoded label tensor.
    """
    image_buffer, label, bbox = parse_example_proto(raw_record)

    image = preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=DEFAULT_IMAGE_SIZE,
        output_width=DEFAULT_IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        cpu_preprocess_fn=cpu_preprocess_fn,
        is_training=is_training,
        seed=seed)
    image = tf.cast(image, dtype)

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    label = tf.cast(
        tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
        dtype=tf.float32)
    return image, label


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):
      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized Example
        protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded':
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label':
            tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update({
        k: sparse_float32 for k in [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax'
        ]
    })

    features = tf.io.parse_single_example(
        serialized=example_serialized, features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

    return features['image/encoded'], label, bbox


def preprocess_image(image_buffer,
                     bbox,
                     output_height,
                     output_width,
                     num_channels,
                     cpu_preprocess_fn=None,
                     is_training=False,
                     seed=None):
    """Preprocesses the given image.
    Preprocessing includes decoding, cropping, and resizing for both training
    and eval images. Training preprocessing, however, introduces some random
    distortion of the image to improve accuracy.
    Args:
      image_buffer: scalar string Tensor representing the raw JPEG image buffer.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as [ymin,
        xmin, ymax, xmax].
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      num_channels: Integer depth of the image buffer for decoding.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
    Returns:
      A preprocessed image.
    """
    if is_training:
        # For training, we want to randomize some of the distortions.
        image = _decode_crop(image_buffer, bbox, num_channels, seed)
        image = _random_horizontal_flip(image, seed)
        image = _resize_image(image, output_height, output_width)
    else:
        # For validation, we want to decode, resize, then just crop the middle.
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method='INTEGER_FAST')
        # The lower bound for the smallest side of the image for aspect-preserving
        # resizing. Originally set to 256 for 224x224 image sizes. Now scaled by the
        # prescribed image size.
        _RESIZE_MIN = int(output_height * float(256) / float(224))
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, output_height, output_width)

    if cpu_preprocess_fn is not None:
        image = cpu_preprocess_fn(image)

    image.set_shape([output_height, output_width, num_channels])

    return image


def _decode_crop(image_buffer, bbox, num_channels, seed=None):
    """Crops the given image to a random part of the image, and randomly flips.
    We use the fused decode_and_crop op, which performs better than the two ops
    used separately in series, but note that this requires that the image be
    passed in as an un-decoded string Tensor.
    Args:
      image_buffer: scalar string Tensor representing the raw JPEG image buffer.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as [ymin,
        xmin, ymax, xmax].
      num_channels: Integer depth of the image buffer for decoding.
    Returns:
      3-D tensor with cropped image.
    """
    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True,
        seed=seed)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.decode_and_crop_jpeg(
        image_buffer, crop_window, channels=num_channels, dct_method='INTEGER_FAST')

    return cropped


def _random_horizontal_flip(image, seed=None):
    return tf.image.random_flip_left_right(image, seed=seed)


def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.
    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.
    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.
    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.compat.v1.image.resize(
        image, [height, width],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)


def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
    Args:
      image: A 3-D image `Tensor`.
      resize_min: A python integer or scalar `Tensor` indicating the size of the
        smallest side after resize.
    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of the
        smallest side after resize.
    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.
    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0],
                    [crop_height, crop_width, -1])
