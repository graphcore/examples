# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

from functools import partial
from typing import Callable, Tuple

import tensorflow as tf


def load_and_preprocess_data(img_path: str, img_width: int, img_height: int,
                             preprocess_fn: Callable, dtype: tf.DType) -> tf.Tensor:
    """Read and pre-process image.

    Args:
        img_path: Path to image
        img_width: Target width
        img_height: Target height
        preprocess_fn: Function that scales the input to the correct range.

    Returns: tf.Tensor representing pre-processed image in fp16.

    """
    image = tf.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = preprocess_fn(image, data_format='channels_last')
    return tf.cast(image, dtype)


def get_dataset(image_filenames: Tuple, batch_size: int, preprocess_fn: Callable, img_width: int, img_height: int,
                loop: bool, dtype: tf.DType) -> tf.data.Dataset:
    """Creates an `Iterator` for enumerating the elements of this dataset.

    Note: The returned iterator will be in an uninitialized state,
    and you must run the `iterator.initializer` operation before using it:

    ```python
    dataset = ...
    iterator = dataset.make_initializable_iterator()
    # ...
    sess.run(iterator.initializer)
    ```

    Args:
        image_filenames: Tuple of image filenames, with each filename corresponding to the label of the image.
        batch_size: Number of images per batch
        preprocess_fn: Pre-processing to apply
        img_width: Expected width of image
        img_height: Expected height of image
        loop: Repeatedly loop through images.
        dtype: Input data type.


    Returns:
        Iterator over images and labels.

    """

    image_ds = tf.data.Dataset.from_tensor_slices(tf.constant([str(item) for item in image_filenames]))
    if loop:
        image_ds = image_ds.repeat()
    input_preprocess = partial(load_and_preprocess_data, img_width=img_width, img_height=img_height,
                               preprocess_fn=preprocess_fn, dtype=dtype)
    image_ds = image_ds.map(map_func=input_preprocess, num_parallel_calls=100)
    image_ds = image_ds.batch(batch_size, drop_remainder=True)
    image_ds = image_ds.prefetch(buffer_size=100)
    return image_ds
