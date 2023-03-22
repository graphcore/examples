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
import enum
import json
import logging
import os
import re
import statistics
import sys
import warnings
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Text,
    Tuple,
    Union,
)

import tensorflow as tf
import tensorflow.keras.backend as K
import yaml
from tensorflow.python import ipu

from hparams_config import Config
from tf2 import efficientdet_keras


logging.basicConfig(level=logging.INFO, format="%(message)s")


class ConfigJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, tf.DType) or isinstance(obj, enum.Enum):
            return obj.name
        else:
            return super().default(obj)


def create_app_json(args: argparse.ArgumentParser, config: Config):
    if args.profile_dir is not None:
        arg_vars = vars(args)

        out_dict = {"arguments": arg_vars, "config": config.as_dict()}

        with open(os.path.join(args.profile_dir, "app.json"), "w") as fh:
            json.dump(out_dict, fh, cls=ConfigJsonEncoder)


def load_weights_into_model(
    args: argparse.Namespace, model: tf.keras.Model, fp32_weights: Optional[List[tf.Tensor]] = None
):
    if args.model_precision != tf.float16:
        model.load_weights(tf.train.latest_checkpoint(args.model_dir))
    else:
        fp16_weights = [w.astype(K.floatx()) for w in fp32_weights]
        model.set_weights(fp16_weights)


def preload_fp32_weights(config: Config, in_shape: Tuple, model_dir: Text) -> List[tf.Tensor]:
    """Create FP32 model, build it and load the latest checkpoint"""
    K.set_floatx("float32")
    fp32_model = efficientdet_keras.EfficientDetNet(config=config)
    fp32_model.build(in_shape)

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    fp32_model.load_weights(latest_checkpoint)
    fp32_weights = fp32_model.get_weights()
    return fp32_weights


def set_or_add_env(key: Text, value: Any):
    if key in os.environ:
        if isinstance(value, dict):
            d = json.loads(os.environ[key])
            os.environ[key] = json.dumps({**d, **value})
        else:
            os.environ[key] += value
    else:
        os.environ[key] = json.dumps(value)


def extract_class_box_outputs(step_outputs: Iterable[tf.Tensor]) -> Tuple[Iterable[tf.Tensor], Iterable[tf.Tensor]]:
    """IPU Keras flattens nested lists to a single list, so we need to manually split
    the outputs back out into classification and bbox outputs"""
    class_outputs = step_outputs[:5]
    box_outputs = step_outputs[5:]
    return class_outputs, box_outputs


def safe_mean(values: Iterable[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.mean(values)
