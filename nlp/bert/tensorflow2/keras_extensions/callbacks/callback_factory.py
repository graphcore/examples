# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import popdist.tensorflow
import tensorflow as tf
from tensorflow.python import ipu

from keras_extensions.callbacks.allreduce_metrics_callback import AllReduceMetricsCallback
from keras_extensions.callbacks.batch_statistics_callback import BatchStatisticsCallback
from keras_extensions.callbacks.checkpoint_callback import CheckpointCallback
from keras_extensions.callbacks.compilation_time_callback import CompilationTimeCallback
from keras_extensions.callbacks.custom_wandb_callback import CustomWandbCallback
from keras_extensions.callbacks.logging_callback import LoggingCallback
from keras_extensions.callbacks.outfeed_queue_callback import OutFeedQueueCallback


class CallbackFactory:
    @staticmethod
    def get_callbacks(
        universal_run_name,
        batch_config,
        model,
        checkpoint_path,
        ckpt_every_n_steps_per_execution,
        outfeed_queues=None,
        distributed_training=False,
        enable_wandb=False,
    ):

        callbacks = []
        log_period = batch_config.steps_per_execution

        logging.info("Creating callback to report batch statistics")
        callbacks.append(BatchStatisticsCallback(batch_config))

        logging.info("Creating callback to report compilation time")
        callbacks.append(CompilationTimeCallback())

        if outfeed_queues is not None:
            logging.info(f"Creating callbacks to read outfeed queues: {outfeed_queues}")
            for outfeed_queue in outfeed_queues:
                callbacks.append(OutFeedQueueCallback(outfeed_queue))

        # For distributed validation perform all reduce on metrics
        if distributed_training:
            callbacks.append(AllReduceMetricsCallback())

        if log_period > 0:
            logging.info(
                "Creating callback for logging to terminal with" f" a period of every {log_period} micro batches"
            )
            callbacks.append(LoggingCallback(log_period=log_period))

            if enable_wandb and popdist.getInstanceIndex() == 0:
                logging.info(
                    "Creating callback for logging to weights and biases"
                    f" a period of every {log_period} micro batches"
                )
                callbacks.append(CustomWandbCallback(log_period=log_period, model=model))
        if popdist.getInstanceIndex() == 0:
            logging.info(
                "Creating callback for creating checkpoints. Checkpoints" f" will be saved to path: {checkpoint_path}"
            )
            callbacks.append(
                CheckpointCallback(
                    universal_run_name=universal_run_name,
                    checkpoint_dir=checkpoint_path,
                    ckpt_every_n_steps_per_execution=ckpt_every_n_steps_per_execution,
                    batch_config=batch_config,
                )
            )

        return callbacks
