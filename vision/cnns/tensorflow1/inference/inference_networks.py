# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

import sys
import tarfile
import urllib.request
from functools import partial
from pathlib import Path
from typing import Dict

import tensorflow.compat.v1 as tf
from models.official_keras.densenet_base import DenseNet
from models.official_keras.inceptionv1_base import InceptionV1
from models.official_keras.inceptionv3_base import InceptionV3
from models.official_keras.mobilenet_base import MobileNet
from models.official_keras.mobilenetv2_base import MobileNetV2
from models.official_keras.nasnet_mobile_base import NASNetMobile
from models.official_keras.resnet50_base import ResNet50
from models.official_keras.xception_base import Xception

from inference_network_base import InferenceNetwork


class DenseNet121Infer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.densenet.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.densenet.decode_predictions

    def build_graph(self, config: Dict):
        # Set up the graph
        model = DenseNet(blocks=config['blocks'],
                         num_classes=self.num_outputs,
                         image_width=self.input_shape[1],
                         image_height=self.input_shape[0],
                         image_channels=self.input_shape[2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class ResNet50Infer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.resnet.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.resnet.decode_predictions

    def build_graph(self, config: Dict):
        model = ResNet50(num_classes=self.num_outputs,
                         image_width=config['input_shape'][1],
                         image_height=config['input_shape'][0],
                         image_channels=config['input_shape'][2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class XceptionInfer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.xception.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.xception.decode_predictions

    def build_graph(self, config: Dict):
        model = Xception(num_classes=self.num_outputs,
                         image_width=config['input_shape'][1],
                         image_height=config['input_shape'][0],
                         image_channels=config['input_shape'][2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class NASNetMobileInfer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.nasnet.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.nasnet.decode_predictions

    def build_graph(self, config: Dict):
        model = NASNetMobile(num_classes=self.num_outputs,
                             image_width=config['input_shape'][1],
                             image_height=config['input_shape'][0],
                             image_channels=config['input_shape'][2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class InceptionV3Infer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.inception_v3.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.inception_v3.decode_predictions

    def build_graph(self, config: Dict):
        model = InceptionV3(num_classes=self.num_outputs,
                            image_width=config['input_shape'][1],
                            image_height=config['input_shape'][0],
                            image_channels=config['input_shape'][2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class MobileNetInfer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.mobilenet.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.mobilenet.decode_predictions

    def build_graph(self, config: Dict):
        model = MobileNet(alpha=config['alpha'],
                          num_classes=self.num_outputs,
                          image_width=config['input_shape'][1],
                          image_height=config['input_shape'][0],
                          image_channels=config['input_shape'][2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class MobileNetV2Infer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.mobilenet_v2.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.mobilenet_v2.decode_predictions

    def build_graph(self, config):
        model = MobileNetV2(num_classes=self.num_outputs,
                            image_width=config['input_shape'][1],
                            image_height=config['input_shape'][0],
                            image_channels=config['input_shape'][2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class InceptionV1Infer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return tf.keras.applications.inception_v3.preprocess_input

    @staticmethod
    def decode_method():
        return tf.keras.applications.inception_v3.decode_predictions

    def build_graph(self, config):
        model = InceptionV1(num_classes=self.num_outputs,
                            image_width=config['input_shape'][1],
                            image_height=config['input_shape'][0],
                            image_channels=config['input_shape'][2])
        output = model.build_model(self.image_input)
        return output.graph, [output.op.name]


class EfficientNetInfer(InferenceNetwork):

    @staticmethod
    def preprocess_method():
        return partial(tf.keras.applications.imagenet_utils.preprocess_input, mode='torch')

    @staticmethod
    def decode_method():
        return tf.keras.applications.imagenet_utils.decode_predictions

    def build_graph(self, config):
        filename, headers = urllib.request.urlretrieve(config['model_url'], f'{config["model"]}.tar.gz')
        with tarfile.open(filename) as tar:
            tar.extractall()
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(
                export_dir=f'{config["model"]}/saved_model',
                sess=sess, tags=[tf.saved_model.tag_constants.SERVING])
            graph_def = tf.get_default_graph().as_graph_def()
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                            graph_def,
                                                                            ["Softmax"])
        return frozen_graph_def, ["Softmax"]


model_dict = {"mobilenet": MobileNetInfer,
              "mobilenetv2": MobileNetV2Infer,
              "nasnetmobile": NASNetMobileInfer,
              "densenet121": DenseNet121Infer,
              "resnet50": ResNet50Infer,
              "xception": XceptionInfer,
              "inceptionv3": InceptionV3Infer,
              "googlenet": InceptionV1Infer,
              "inceptionv1": InceptionV1Infer,
              "efficientnet-s": EfficientNetInfer,
              "efficientnet-m": EfficientNetInfer,
              "efficientnet-l": EfficientNetInfer,
              }
