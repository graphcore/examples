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
"""Helper utility to download weights for image classification models trained on imagenet."""

import tarfile
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow import pywrap_tensorflow
from tensorflow.compat.v1.keras import backend as keras_backend
from tensorflow.compat.v1.keras.applications.densenet import DenseNet121
from tensorflow.compat.v1.keras.applications.inception_v3 import InceptionV3
from tensorflow.compat.v1.keras.applications.mobilenet import MobileNet
from tensorflow.compat.v1.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.compat.v1.keras.applications.nasnet import NASNetMobile
from tensorflow.compat.v1.keras.applications.resnet import ResNet50
from tensorflow.compat.v1.keras.applications.xception import Xception
from tensorflow.compat.v1.keras.utils import get_file


def convert_ckpt_to_fp16(ckpt_file: str) -> tf.train.Saver:
    """Convert checkpoint to fp16 weights and return saver.

    Args:
        ckpt_file: Path to checkpoint file.

    Returns: tf.train.Saver object initialized with dictionary of fp16 variables.

    """
    # Strip .data-xxxx-xxxx
    if not ckpt_file.endswith(".ckpt"):
        ckpt_file = ckpt_file.rsplit('.', 1)[0]

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_map = reader.get_variable_to_shape_map()

    val_f16 = {}
    for key, _ in var_to_map.items():
        val_f16[key.strip(":0")] = tf.Variable(reader.get_tensor(key).astype(np.float16))
    saver = tf.train.Saver(val_f16)
    return saver


def get_weights(save_dir: Path, model_name: str, dtype: str) -> str:
    """Download pre-trained imagenet weights for model.

    Args:
        save_dir: Path to where checkpoint must be downloaded.
        model_name: Type of image classification model, must be one of
        ("GoogleNet", "InceptionV1", "MobileNet", "MobileNetV2", "NASNetMobile", "DenseNet121",
         "ResNet50", "Xception", "InceptionV3") in all lower case.
        dtype: Data type of the network.

    Returns: Path to checkpoint file.

    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        keras_backend.set_floatx(dtype)
        keras_backend.set_session(sess)
        if model_name == "mobilenet":
            MobileNet(weights='imagenet')
            saver = tf.train.Saver()
        elif model_name == "mobilenetv2":
            MobileNetV2(weights='imagenet')
            saver = tf.train.Saver()
        elif model_name == "nasnetmobile":
            NASNetMobile(weights='imagenet')
            saver = tf.train.Saver()
        elif model_name == "densenet121":
            DenseNet121(weights='imagenet')
            saver = tf.train.Saver()
        elif model_name == "resnet50":
            ResNet50(weights='imagenet')
            saver = tf.train.Saver()
        elif model_name == "xception":
            Xception(weights='imagenet')
            saver = tf.train.Saver()
        elif model_name == "inceptionv3":
            InceptionV3(weights='imagenet')
            saver = tf.train.Saver()
        elif model_name in ("googlenet", "inceptionv1"):
            # This download can cause issues on the CI, as it needs to create ~/.keras directory
            # One way to solve that is to copy ~/.keras on the CI
            tar_file = get_file(fname='inceptionv1_tar.gz',
                                origin='http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz')
            tar_file_reader = tarfile.open(tar_file)
            tar_file_reader.extractall(save_dir)
            if dtype == 'float16':
                saver = convert_ckpt_to_fp16(Path(save_dir, 'inception_v1.ckpt').as_posix())
            sess.run(tf.global_variables_initializer())
        else:
            raise ValueError("""Requested model type = %s not one of
            ["GoogleNet", "InceptionV1", "MobileNet", "MobileNetV2", "NASNetMobile", "DenseNet121",
            "ResNet50", "Xception", "InceptionV3"].""" % model_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        return saver.save(sess, Path(save_dir, f"{model_name}.ckpt").as_posix())
