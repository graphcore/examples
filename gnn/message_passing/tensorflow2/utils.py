# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import time
import wandb

import tensorflow as tf
from absl import logging
from tensorflow import keras

import xpu


class ThroughputCallback(keras.callbacks.Callback):
    def __init__(self, samples_per_epoch, log_wandb=True):
        super().__init__()
        self.time = 0
        self.samples_per_epoch = samples_per_epoch
        self.log_wandb = log_wandb

    def on_epoch_begin(self, epoch, logs=None):
        self.time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_per_epoch = time.time() - self.time
        samples_per_sec = self.samples_per_epoch / time_per_epoch
        if self.log_wandb:
            wandb.log({"throughput": samples_per_sec})

        logging.info(f"\nthroughput: {samples_per_sec:.2f} samples/sec")


def get_optimizer(
    name="adam",
    learning_rate=1e-5,
    dtype="float32",
    m_dtype=None,
    v_dtype=None,
    gradient_accumulation_factor=1,
    replicas=1,
):
    def rescale_gradients(grads_and_vars):
        return [(g / (gradient_accumulation_factor * replicas), v) for g, v in grads_and_vars]

    if name == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, gradient_transformers=[rescale_gradients])
    elif name == "adam" and dtype == tf.float32:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, gradient_transformers=[rescale_gradients])
    elif name == "adam" and dtype == tf.float16:
        opt = xpu.AdamIpuOptimizer(
            learning_rate=learning_rate, gradient_transformers=[rescale_gradients], m_dtype=m_dtype, v_dtype=v_dtype
        )
    else:
        raise NotImplementedError(f"Optimizer {name} is not supported.")

    return opt


def size_hr(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def print_trainable_variables(model):
    logging.info("Trainable Variables:")
    total_parameters = 0
    total_size = 0
    for variable in model.trainable_variables:
        variable_parameters = 1
        for DIM in variable.shape:
            variable_parameters *= DIM
        variable_size = variable_parameters * variable.dtype.size
        logging.info(f"{variable.name}, {variable.shape}, {variable.dtype} ({size_hr(variable_size)})")
        total_parameters += variable_parameters
        total_size += variable_size
    logging.info(f"Total Parameters: {total_parameters:,}  ({size_hr(total_size)})")


def str_dtype_to_tf_dtype(str_dtype):
    return {"float16": tf.float16, "float32": tf.float32}[str_dtype]
