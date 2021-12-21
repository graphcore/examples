# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python import ipu
from typing import List, Tuple, Optional

from .throughput_estimator_callback import ThroughputEstimatorCallback
from .compilation_time_callback import CompilationTimeCallback
from .logging_callback import LoggingCallback
from .custom_wandb_callback import CustomWandbCallback
from .outfeed_queue_callback import OutFeedQueueCallback
from .checkpoint_callback import CheckpointCallback
from .allreduce_metrics_callback import AllReduceMetricsCallback
from .cpu_memory_callback import CPUMemoryCallback
from .data_remover_callback import DataRemoverCallback


class CallbackFactory:

    @staticmethod
    def get_callbacks(wandb: bool,
                      log_period: int,
                      steps_per_execution: int,
                      micro_batch_size: int,
                      model: tf.keras.Model,
                      outfeed_queues: Optional[List[Tuple[str, ipu.ipu_outfeed_queue.IPUOutfeedQueue]]] = None,
                      checkpoints: bool = False,
                      checkpoint_dir: str = '/tmp/checkpoints',
                      distributed_training: bool = False,
                      args: dict = {},
                      fields_to_remove: List[str] = []):

        callbacks = []

        # Add metric callbacks

        callbacks.append(ThroughputEstimatorCallback(steps_per_execution=steps_per_execution,
                                                     micro_batch_size=micro_batch_size))
        callbacks.append(CompilationTimeCallback())

        callbacks.append(CPUMemoryCallback())

        if outfeed_queues is not None:
            for name, outfeed_queue in outfeed_queues:
                callbacks.append(OutFeedQueueCallback(queue=outfeed_queue, name=name))

        if len(fields_to_remove) > 0:
            callbacks.append(DataRemoverCallback(fields_to_remove))

        if checkpoints:
            callbacks.append(CheckpointCallback(ckpt_period=log_period,
                                                checkpoint_dir=checkpoint_dir))

        # For distributed validation peform all reduce on metrics
        if distributed_training:
            callbacks.append(AllReduceMetricsCallback())

        # Add log callbacks
        logging_log_period = log_period
        if logging_log_period > 0:
            callbacks.append(LoggingCallback(log_period=logging_log_period))
        if wandb:
            wandb_log_period = log_period
            if wandb_log_period > 0:
                callbacks.append(CustomWandbCallback(log_period=wandb_log_period,
                                                     args=args,
                                                     model=model))

        return callbacks
