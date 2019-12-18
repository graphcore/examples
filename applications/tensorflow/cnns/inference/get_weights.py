# Copyright 2019 Graphcore Ltd.
"""Helper utility to download weights for image classification models trained on imgenet."""
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow import pywrap_tensorflow
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.utils import get_file


def convert_ckpt_to_fp16(ckpt_file: str) -> tf.compat.v1.train.Saver:
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
        elif model_name in ("googleNet", "inceptionv1"):
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
