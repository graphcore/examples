# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import time
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.ipu import config as ipu_config


def create_ipu_config(
        available_memory_proportion,
        num_required_ipus,
        partials_type="half",
        fp_exceptions=False,
        enable_stochastic_rounding=True,
        num_io_tiles=0):
    cfg = ipu_config.IPUConfig()
    if num_io_tiles > 0:
        cfg.io_tiles.num_io_tiles = num_io_tiles
        cfg.io_tiles.place_ops_on_io_tiles = True
    cfg.allow_recompute = True
    cfg.auto_select_ipus = num_required_ipus
    cfg.selection_order = ipu_config.SelectionOrder.AUTO

    cfg.convolutions.poplar_options = {
        "availableMemoryProportion": str(available_memory_proportion),
        "partialsType": partials_type
    }
    cfg.matmuls.poplar_options = {
        "availableMemoryProportion": str(available_memory_proportion),
        "partialsType": partials_type
    }

    cfg.experimental.always_rearrange_copies_on_the_host = False
    # FP behaviou
    cfg.floating_point_behaviour.inv = fp_exceptions
    cfg.floating_point_behaviour.div0 = fp_exceptions
    cfg.floating_point_behaviour.oflo = fp_exceptions
    cfg.floating_point_behaviour.nanoo = fp_exceptions
    cfg.floating_point_behaviour.esr = ipu_config.StochasticRoundingBehaviour.from_bool(enable_stochastic_rounding)
    cfg.norms.use_stable_statistics = True

    # optimizations
    cfg.optimizations.enable_graph_outlining = True
    cfg.optimizations.merge_infeed_io_copies = True
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = 10*1024*1024

    cfg.configure_ipu_system()
    return cfg


class ThroughputCallback(keras.callbacks.Callback):
    def __init__(self, samples_per_epoch):
        super().__init__()
        self.epoch_time = 0.0
        self.samples_per_epoch = samples_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time = time.time() - self.epoch_time
        tput = self.samples_per_epoch / self.epoch_time
        print(
            "Duration: {:.2f}s, throughput: {:.2f} samples/sec".format(self.epoch_time, tput))
        logs = logs or {}
        logs.update({'throughput': tput, 'duration': self.epoch_time})
        super().on_epoch_end(epoch, logs)


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, epochs_per_save=1, max_to_keep=None, **kwargs):
        super().__init__(save_freq="epoch", **kwargs)
        self.epoch_count = 0
        self.max_to_keep = max_to_keep
        self.epochs_per_save = epochs_per_save
        self.filepath = kwargs["filepath"]

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epochs_per_save == 0:
            super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        # save last checkpoint
        self.model.save_weights(
            filepath=os.path.dirname(self.filepath)+"/model.h5")
        super().on_train_end(logs)


class LearningRateLogger(keras.callbacks.Callback):
    def __init__(self, steps_per_epoch=1):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            logs.update(
                {'learning_rate': self.model.optimizer._decayed_lr(tf.float32).numpy()})
        except:
            cur_steps = int(epoch * self.steps_per_epoch)
            logs.update(
                {
                    'Steps': cur_steps,
                    'Epoch': epoch,
                    'learning_rate': self.model.optimizer.learning_rate(cur_steps).numpy()})
        super().on_epoch_end(epoch, logs)


class CompilationTimeCallback(keras.callbacks.Callback):
    def __init__(self):
        self.__current_batch_operations = self.__calculate_compilation_time

    def on_train_begin(self, logs=None):
        self.compilation_start_time = time.time()

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.__current_batch_operations(logs)

    def __do_nothing(self, logs):
        pass

    def __calculate_compilation_time(self, logs):
        if logs is not None:
            comp_time = time.time() - self.compilation_start_time
            logs['Compilation Time'] = comp_time
            print(f"Compilation Time: {comp_time}s")
        self.__current_batch_operations = self.__do_nothing
