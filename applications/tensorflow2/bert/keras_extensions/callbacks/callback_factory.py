# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.python import ipu

from data_utils.batch_config import BatchConfig
from keras_extensions.callbacks.batch_statistics_callback import BatchStatisticsCallback
from keras_extensions.callbacks.checkpoint_callback import CheckpointCallback
from keras_extensions.callbacks.compilation_time_callback import CompilationTimeCallback
from keras_extensions.callbacks.custom_wandb_callback import CustomWandbCallback
from keras_extensions.callbacks.logging_callback import LoggingCallback
from keras_extensions.callbacks.outfeed_queue_callback import OutFeedQueueCallback


class CallbackFactory:

    @staticmethod
    def get_callbacks(universal_run_name: str,
                      batch_config: BatchConfig,
                      model: tf.keras.Model,
                      checkpoint_path: Path,
                      ckpt_every_n_steps_per_execution: int,
                      outfeed_queues: Optional[List[Tuple[str, ipu.ipu_outfeed_queue.IPUOutfeedQueue]]] = None,
                      config: dict = {}):

        callbacks = []
        log_period = batch_config.steps_per_execution

        print("Creating callback to report batch statistics")
        callbacks.append(BatchStatisticsCallback(batch_config))

        print("Creating callback to report compilation time")
        callbacks.append(CompilationTimeCallback())

        if outfeed_queues is not None:
            print(f"Creating callbacks to read outfeed queues: {outfeed_queues}")
            for outfeed_queue in outfeed_queues:
                callbacks.append(OutFeedQueueCallback(outfeed_queue))

        if log_period > 0:
            print("Creating callback for logging to terminal with"
                  f" a period of every {log_period} micro batches")
            callbacks.append(LoggingCallback(log_period=log_period))

            if config["wandb_opts"]["log_to_wandb"]:
                print("Creating callback for logging to weights and biases"
                      f" a period of every {log_period} micro batches")
                callbacks.append(CustomWandbCallback(name=universal_run_name,
                                                     log_period=log_period,
                                                     config=config,
                                                     model=model))

        print("Creating callback for creating checkpoints. Checkpoints"
              f" will be saved to path: {checkpoint_path}")
        callbacks.append(CheckpointCallback(universal_run_name=universal_run_name,
                                            checkpoint_dir=checkpoint_path,
                                            ckpt_every_n_steps_per_execution=ckpt_every_n_steps_per_execution,
                                            batch_config=batch_config))

        return callbacks
